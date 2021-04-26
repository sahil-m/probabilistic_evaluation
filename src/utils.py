import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing._label import _encode
from sklearn.utils import column_or_1d
import math
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler

##### For ROC_AUC calculation
def get_predictions_from_nb_model(ppc_idata, df_data: pd.DataFrame, range_for_probabilities: int = 1000, target='onrent'):  
    # generate the probabilistic prediction from negative binomial model and store them in the dataframe.
    output_probabilistic = ppc_idata.posterior_predictive.obs.to_dataframe()
    df_data[target + '_pred_mean'] = output_probabilistic.mean(level=2).to_numpy().flatten().astype(int)
    df_data[target + '_pred_stddev'] = output_probabilistic.std(level=2).to_numpy().flatten().astype(int)
    df_data[target + '_plus_1_stddev'] = df_data[target + '_pred_mean'] + df_data[target + '_pred_stddev']
    df_data[target + '_minus_1_stddev'] = df_data[target + '_pred_mean'] - df_data[target + '_pred_stddev']
    df_data[target + '_plus_2_stddev'] = df_data[target + '_pred_mean'] + 2*df_data[target + '_pred_stddev']
    df_data[target + '_minus_2_stddev'] = df_data[target + '_pred_mean'] - 2*df_data[target + '_pred_stddev']
    df_data[target + '_minus_1_stddev'] = df_data[target + '_minus_1_stddev'].apply(lambda x: x if x >=0 else 0)
    df_data[target + '_minus_2_stddev'] = df_data[target + '_minus_2_stddev'].apply(lambda x: x if x >=0 else 0)    
    result_matrix = output_probabilistic.mean(level=(1, 2)).reset_index().pivot(index='obs_dim_0', columns='draw', values='obs').to_numpy()
    onrent_pred_dist = result_matrix.tolist()
    df_data[target + '_pred_dist'] = onrent_pred_dist
    
    for i in range(0, range_for_probabilities):
        df_data['prob_' + str(i)] = result_matrix[:,i]
    return df_data

def create_bins_using_quantiles(num_bins: int, values_to_bin: np.array) -> list:
    
    range_min = values_to_bin.min()
    range_max = values_to_bin.max()
    bins = []
    
    current_bin_min = 0
    if range_min > 0:
        bins.append((0,range_min-1))
        num_bins = num_bins -1 
        current_bin_min = range_min

    list_quantiles = np.arange(0,1, 1/num_bins)
    list_quantiles_values = np.quantile(a=values_to_bin, q=list_quantiles, interpolation='nearest')

    for i in range(len(list_quantiles_values)-1):
        if (list_quantiles_values[i+1] > list_quantiles_values[i]) :
            bins.append((list_quantiles_values[i], list_quantiles_values[i+1]-1))
    
    # Append last bin from last quantile to max value as we have not traversed the whole list in above loop. 
    bins.append((list_quantiles_values[-1] , range_max))
    return bins

def create_bins_of_equal_ranges(num_bins: int, values_to_bin: np.array) -> list:
    
    range_min = values_to_bin.min()
    range_max = values_to_bin.max()
    bins = []
    
    current_bin_min = 0
    if range_min > 0:
        bins.append((0,range_min-1))
        num_bins = num_bins -1 
        current_bin_min = range_min

    bin_size = math.ceil((range_max - range_min) / num_bins) 
    current_bin_max = current_bin_min + bin_size - 1
    while current_bin_max <= (range_max + bin_size):
        bins.append((current_bin_min,current_bin_max))
        current_bin_min = current_bin_min + bin_size 
        current_bin_max = current_bin_min + bin_size -1
    
    return bins

def calculate_auc_for_probability_outputs_v1(df: pd.DataFrame, target : str, num_bins: int = 10, binning: str = 'equal') -> (float, pd.DataFrame) :
    # binning can take two different arguments - 'equal' & 'quantile'
    # have one bin for the last range which would be some value --> infinity
    num_bins = num_bins - 1 
    
    if binning == 'quantile':
        bins = create_bins_using_quantiles(num_bins=num_bins, values_to_bin=df[target].to_numpy())
    else:
        bins = create_bins_of_equal_ranges(num_bins=num_bins, values_to_bin=df[target].to_numpy())

    onrents = []  # container for actual label
    classes = []  # container for predicted label
    unique_labels = [] # container for unique labels 
    all_prob_cols = []
    for range_bin in bins:
        label_this = "onrents_[{},{}]".format(range_bin[0], range_bin[1])
        unique_labels.append(label_this)
        cols_to_sum = ["prob_" + str(i) for i in range(range_bin[0], range_bin[1]+1)]
        all_prob_cols = all_prob_cols + cols_to_sum
        df[label_this] = df[cols_to_sum].sum(axis=1)
        for j in range(range_bin[0], range_bin[1]+1):
            onrents.append(j)
            classes.append(label_this)

    label_this = "onrents_[{},{}]".format(bins[-1][1]+1, 'infinity')
    unique_labels.append(label_this)
    df[label_this] = 1 - df[all_prob_cols].sum(axis=1)
    target = 'onrent'

    df_onrent_classes = pd.DataFrame({target: onrents, 'Label': classes})
    df = df.merge(df_onrent_classes, how="left", on=target)
    probs_onrents = [s for s in list(df.columns) if "onrents" in s]
    
    predefined_list = [target, target + '_pred_mean', target + '_pred_stddev', target + '_pred_dist']
    probs_onrents = [x for x in probs_onrents if x not in predefined_list]
    # sort labels for usage as class labels with AUC of ROC computation
    df['Predicted'] = df[probs_onrents].idxmax(axis=1)
    
    labels = column_or_1d(probs_onrents)
    classes_these = _encode(labels)
    probs_onrents = classes_these
    
    auc = roc_auc_score(y_true=df["Label"],
                        y_score=df[probs_onrents],
                        multi_class='ovo',
                        labels=probs_onrents
                        )
    
    df_confusion_matrix = pd.DataFrame(data=confusion_matrix(y_true=df['Label'], y_pred=df['Predicted'], 
                                                               labels=unique_labels), index=unique_labels, 
                                       columns=unique_labels)
    return auc, df_confusion_matrix

##### Rank based metric: Weighted average probability-mass-around-true-based interval rank
def get_bins(df_pred, target='onrent', n_bins=20):  
    mean_std = int(round(df_pred[target + '_pred_stddev'].mean(), 0))
    df_pred[target + '_pred_stddev'] = mean_std
    bin_size = int(round((6 * mean_std)/n_bins, 0)) # because for normal +/- 3 std covers 99.7% prob. mass
    if bin_size % 2 != 0:
        bin_size += 1
    
    for i in (np.arange(n_bins)+1):
        df_temp = pd.DataFrame()
        df_temp['lower'] = df_pred[target] - i*int(bin_size/2)
        df_temp['higher'] = df_pred[target] + i*int(bin_size/2)
        col_name = '{}_bin_{:02d}'.format(target, i)
        df_pred[col_name] = df_temp.values.tolist()
    
    # col_name = '{}_bin_{:02d}'.format(target, (n_bins+1))
    # df_pred[col_name] =         
    
    return df_pred

def get_prob_mass_from_normal(dfr, bin_col_name, target='onrent'):
    norm_ = norm(dfr[target + '_pred_mean'], dfr[target + '_pred_stddev'])
    range_bin = dfr[bin_col_name]
    return norm_.cdf(range_bin[1]) - norm_.cdf(range_bin[0])

def get_prob_mass(df, target='onrent'):
    bin_col_names = df.filter(regex=target + '_bin_').columns
    for bin_col_name in bin_col_names:
        req_cols = [target + '_pred_mean', target + '_pred_stddev', bin_col_name]
        df[bin_col_name+'_prob_cum'] = df[req_cols].apply(get_prob_mass_from_normal, axis=1, bin_col_name=bin_col_name, target=target)
    return df

def get_pred_dist(df, target='onrent'):
    cum_prob_col_names = df.filter(regex=target + '_bin_[0-9]+_prob_cum').columns.sort_values()
    for i, col_name in enumerate(cum_prob_col_names):
        if i == 0:
            df[col_name.replace('_cum', '')] = df[col_name]
        else:
            df[col_name.replace('_cum', '')] = df[col_name] -  df[cum_prob_col_names[i-1]]
    return df.drop(columns=cum_prob_col_names)

def get_weigAv_prob_rank(df_pred, target='onrent', n_bins=20):
    df_pred = get_bins(df_pred, target, n_bins)
    df_pred = get_prob_mass(df_pred, target)
    df_pred = get_pred_dist(df_pred, target)
    
    # get metric from probs data
    prob_col_names = df_pred.filter(regex=target + '_bin_[0-9]+_prob').columns
    sum_probsOverObs = df_pred[prob_col_names].sum(axis=0)
    prob_mass_in_ranges = sum_probsOverObs/df_pred.shape[0]
    # print(prob_mass_in_ranges)
    print('prob. mass covered: ', prob_mass_in_ranges.sum())
    weigAv_prob_rank = (prob_mass_in_ranges * (np.arange(len(prob_mass_in_ranges))+1)).sum()
    return weigAv_prob_rank

##### Rank based metric: Weighted average predicted HDI interval rank
def get_hdi_range(posterior_predictive_obs, hdi_prob, scaler):
    hdi_xr = az.hdi(posterior_predictive_obs, hdi_prob=hdi_prob)
    hdi_xr = xr.apply_ufunc(scaler.inv_transform, hdi_xr)
    hdi_df = hdi_xr.to_dataframe().reset_index(level=0).pivot(columns='hdi', values='obs').round(0).astype(int)
    hdi_df = hdi_df[['lower', 'higher']]
    return hdi_df.values.tolist()

def get_all_hdi_ranges(posterior_predictive_obs, scaler, hdi_probs=None):
    if hdi_probs is None:
        hdi_probs = np.round(np.arange(0, 1, 0.05)[1:], 2)
    df = pd.DataFrame()
    for hdi_prob in hdi_probs:
        col_name = 'range_hdi_prob_{}'.format(hdi_prob,2)
        df[col_name] = get_hdi_range(posterior_predictive_obs, hdi_prob, scaler)
    return df

def get_hdi_ranks(df_hdi_ranges, hdi_probs=None, target='onrent'):
    if hdi_probs is None:
        hdi_probs = np.round(np.arange(0, 1, 0.05)[1:], 2)  
    df_hdi_ranks = pd.DataFrame() 
    for i, hdi_prob in enumerate(hdi_probs):
        rank = i + 1
        rank_col_name = 'rank_{}_hdi_prob_{}'.format(rank, hdi_prob)
        range_col_name = 'range_hdi_prob_{}'.format(hdi_prob)
        df_hdi_ranks[rank_col_name] = df_hdi_ranges.apply(lambda dfr: dfr[target] in range(dfr[range_col_name][0], dfr[range_col_name][1]+1), axis=1)
    return df_hdi_ranks

def get_weigAv_hdi_rank(data, posterior_predictive_obs, scaler, hdi_probs=None):
    if hdi_probs is None:
        hdi_probs = np.round(np.arange(0, 1, 0.05)[1:], 2)

    # get range against hdis for each row
    all_hdi_ranges = get_all_hdi_ranges(posterior_predictive_obs, scaler, hdi_probs)
    
    # add target
    all_hdi_ranges_with_target = pd.concat([data, all_hdi_ranges], axis=1)
    
    # get rank of range in which true value lies
    hdi_ranks = get_hdi_ranks(all_hdi_ranges_with_target, hdi_probs)
    
    # get metric from rank data
    hdi_ranks_sumOverObs = hdi_ranks.sum(axis=0)
    hdi_ranks_freq = hdi_ranks_sumOverObs.diff()
    hdi_ranks_freq[0] = hdi_ranks_sumOverObs[0]
    hdi_ranks_freq_sum = hdi_ranks_freq.sum()
    print('Proportion of observations which are within the range against hdi probability of', max(hdi_probs), ': ', round(hdi_ranks_freq_sum/data.shape[0], 2))
    weigAv_hdi_rank = (hdi_ranks_freq * (np.arange(len(hdi_ranks_freq))+1)).sum()/hdi_ranks_freq_sum
    print(hdi_ranks.filter(regex='rank_{}_'.format(round(weigAv_hdi_rank))).columns)
    return hdi_ranks, hdi_ranks_freq, weigAv_hdi_rank

#####
def get_error_sd_metric(df_pred, target='onrent'):
    df_pred['error'] = df_pred[target] - df_pred[target + '_pred_mean']
    df_pred['error_abs'] = df_pred['error'].abs()
    df_pred['sd_minus_absError'] = df_pred[target + '_pred_stddev'] - df_pred['error_abs']
    df_pred['sd_minus_absError_abs'] = df_pred['sd_minus_absError'].abs()
    df_pred['metric'] = df_pred['sd_minus_absError_abs'] + df_pred['error_abs']
    
    return df_pred['metric'].sum()/df_pred.shape[0]

#####
def get_z_metric(df_pred, target='onrent'):
    df_pred['error'] = df_pred[target] - df_pred[target + '_pred_mean']
    df_pred['error_abs'] = df_pred['error'].abs()
    df_pred['mape'] = df_pred['error_abs']/(df_pred[target] + 1)
    df_pred['z_score'] = df_pred['error_abs']/df_pred[target + '_pred_stddev']
    df_pred['metric'] = -df_pred['z_score'] + df_pred['mape']
    
#     scaler = MinMaxScaler()
#     df_pred['metric_scaled'] = scaler.fit_transform(df_pred['metric'].values.reshape(-1,1))
    
    return df_pred['metric'].sum()/df_pred.shape[0]
