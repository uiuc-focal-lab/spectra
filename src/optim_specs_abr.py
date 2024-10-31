import pandas as pd 
import argparse
import os 
import sys 
sys.path.append('src/')
import common_optim_specs_new as cos
import pickle

all_columns = [
    'Last1_chunk_bitrate', 'Last1_buffer_size', 'Last8_throughput', 'Last7_throughput',
    'Last6_throughput', 'Last5_throughput', 'Last4_throughput', 'Last3_throughput',
    'Last2_throughput', 'Last1_throughput', 'Last8_downloadtime', 'Last7_downloadtime',
    'Last6_downloadtime', 'Last5_downloadtime', 'Last4_downloadtime', 'Last3_downloadtime',
    'Last2_downloadtime', 'Last1_downloadtime', 'chunksize1', 'chunksize2', 'chunksize3',
    'chunksize4', 'chunksize5', 'chunksize6', 'Chunks_left'
]

brs = [300.0,750.0,1200.0,1850.0,2850.0,4300.0]
output_name = 'br'
parser = argparse.ArgumentParser(description='Optimization based specs generation')
parser.add_argument('-fig', action='store_true', help='plot the figures') # DONT CHANGE
parser.add_argument('-slack', type=int, default=0, help='slack in the eps') # DON'T CHANGE
parser.add_argument('-tight', type=int, default=5, help='tightness of the specs') # [1, len(brs)-1]
parser.add_argument('-coverage', type=float, default=1.1, help='coverage of the specs over behavior regions') # default gets max coverage possible # DONT CHANGE
parser.add_argument('-parts', type=int, default=100, help='number of parts to divide each dimension of the input space') # [10, 20, 50, 70, 100]
parser.add_argument('-rep_thresh', type=float, default=0.01, help='representation threshold for each spec') # [0.01, 0.02, 0.05, 0.1]
parser.add_argument('-history', type=int, default=3, help='number of features to consider in history') # [1, 2, 3, 4, 5, 6, 7, 8]
parser.add_argument('-filename', type=str, default=None, help='filename to save the specs')
parser.add_argument('--full_spec', action='store_true', help='generate full specs')
args = parser.parse_args()

features_of_interest = ['Last1_buffer_size'] + [f'Last{i}_downloadtime' for i in range(1, args.history+1)]
features_of_interest = [all_columns.index(f) for f in features_of_interest]
num_feats = len(features_of_interest)
parts = args.parts
rep_thresh = args.rep_thresh # 1% of the data in each spec
min_points_thresh = 1/(parts**num_feats)

coverage = args.coverage # 80% of the behavior regions should be covered by the specs
tight = args.tight # allowing at max 2 bitrates in each spec
zeros = 0
abrs = ['bb', 'mpc'] 
mode = 'nontrivial' # other option is 'nontrivial'
feat_mins = [float('inf')]*num_feats
feat_maxs = [float('-inf')]*num_feats

# read in data and get min/max for each feature
columns = features_of_interest
data_files = {
    'bb': 'data/bb_events_pensieve_train.csv',
    'mpc': 'data/mpc_events_pensieve_train.csv'
}
columns_str = [str(c) for c in columns]
columns_str = '_'.join(columns_str)
data_path = f'data/abr_data_{columns_str}_train.pkl'
if not os.path.exists(data_path):
    abr_data = {}
    for abr in abrs:
        data = {}
        df = pd.read_csv(data_files[abr])        
        df_brs = df[output_name].unique()
        for b in df_brs:
            df_this_br = df[df[output_name] == b]
            data[b] = [[df_this_br.iloc[i][c] for c in columns] for i in range(df_this_br.shape[0])]
        abr_data[abr] = data
    pickle.dump(abr_data, open(data_path, 'wb'))
else:
    abr_data = pickle.load(open(data_path, 'rb'))
for abr in abrs:
    for br in abr_data[abr].keys():
        for point in abr_data[abr][br]:
            assert len(point) == num_feats
            for j in range(num_feats):
                feat_mins[j] = min(feat_mins[j], point[j])
                feat_maxs[j] = max(feat_maxs[j], point[j])
assert feat_mins <= feat_maxs
print('data read in')

cos.spec_fun(abrs, brs, columns, all_columns, data_files, args, feat_mins, feat_maxs, abr_data, 'abr')
