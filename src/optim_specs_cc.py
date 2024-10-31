import pandas as pd 
import argparse
import json 
import os 
import sys 
sys.path.append('src/')
import common_optim_specs_new as cos

all_columns = [
    'latency_gradient_1', 'latency_gradient_2', 'latency_gradient_3', 'latency_gradient_4', 'latency_gradient_5', 'latency_gradient_6', 'latency_gradient_7', 'latency_gradient_8', 'latency_gradient_9', 'latency_gradient_10',
    'latency_ratio_1', 'latency_ratio_2', 'latency_ratio_3', 'latency_ratio_4', 'latency_ratio_5', 'latency_ratio_6', 'latency_ratio_7', 'latency_ratio_8', 'latency_ratio_9', 'latency_ratio_10',
    'sending_ratio_1', 'sending_ratio_2', 'sending_ratio_3', 'sending_ratio_4', 'sending_ratio_5', 'sending_ratio_6', 'sending_ratio_7', 'sending_ratio_8', 'sending_ratio_9', 'sending_ratio_10'
]

csrs = ['+', '0', '-']
output_name = 'change_of_sending_rate'
parser = argparse.ArgumentParser(description='Optimization based specs generation')
parser.add_argument('-fig', action='store_true', help='plot the figures') # DONT CHANGE
parser.add_argument('-slack', type=int, default=0, help='slack in the eps') # DONT CHANGE
parser.add_argument('-tight', type=int, default=2, help='tightness of the specs') # [1, len(csrs)-1]
parser.add_argument('-coverage', type=float, default=1.1, help='coverage of the specs over behavior regions') # default gets max coverage possible # DONT CHANGE
parser.add_argument('-parts', type=int, default=50, help='number of parts to divide each dimension of the input space') # [10, 20, 50, 70, 100] 
parser.add_argument('-rep_thresh', type=float, default=0.01, help='representation threshold for each spec') # [0.01, 0.02, 0.05, 0.1]
parser.add_argument('-history', type=int, default=4, help='number of features to consider in history') # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

parser.add_argument('-filename', type=str, default=None, help='filename to save the specs')
parser.add_argument('--full_spec', action='store_true', help='generate full specs')
args = parser.parse_args()

features_of_interest = [0+i for i in range(args.history)] + [10+i for i in range(args.history)] + [20+i for i in range(args.history)]
num_feats = len(features_of_interest)
parts = args.parts
rep_thresh = args.rep_thresh
min_points_thresh = 1/(parts**num_feats)

coverage = args.coverage
tight = args.tight # allowing at max 2 types of sending_rates in each spec
ccs = ['cubic', 'bbr'] 
mode = 'nontrivial' # only option
feat_mins = [float('inf')]*num_feats
feat_maxs = [float('-inf')]*num_feats

# read in data and get min/max for each feature
columns = features_of_interest
data_files = {
    'bbr': 'data/bbr_events_train_filtered.csv',
    'cubic': 'data/cubic_events_train_filtered.csv'
}
columns_str = [str(c) for c in columns]
columns_str = '_'.join(columns_str)
data_path = f'data/cc_data_{columns_str}_train.json'
if not os.path.exists(data_path):
    cc_data = {}
    for cc in ccs:
        data = {}
        df = pd.read_csv(data_files[cc])
        df_csrs = df[output_name].unique()
        for b in df_csrs:
            df_this_csr = df[df[output_name] == b]
            data[b] = [[df_this_csr.iloc[i][all_columns[c]] for c in columns] for i in range(df_this_csr.shape[0])]
        cc_data[cc] = data
    json.dump(cc_data, open(data_path, 'w'))
else:
    cc_data = json.load(open(data_path, 'r'))
for cc in ccs:
    for csr in cc_data[cc].keys():
        for point in cc_data[cc][csr]:
            assert len(point) == num_feats
            for j in range(num_feats):
                feat_mins[j] = min(feat_mins[j], point[j])
                feat_maxs[j] = max(feat_maxs[j], point[j])
assert feat_mins <= feat_maxs
print('data read in')
# partition input space into regions


cos.spec_fun(ccs, csrs, columns, all_columns, data_files, args, feat_mins, feat_maxs, cc_data, 'cc')
