import pandas as pd 
import json
import numpy as np
import argparse 

def get_accuracy(df_test, abr, feats, vals, return_vals = False):
    br_col = df_test['br'].values
    cols = [df_test[col].values for col in feats[0].keys()]

    # Precompute the condition masks
    masks = []
    for j in range(len(feats)): # going over all specs one-by-one
        feat_masks = []
        for f, (lower, upper) in feats[j].items():
            feat_masks.append(np.logical_and(df_test[f].values >= lower, df_test[f].values <= upper))
        masks.append(np.logical_and.reduce(feat_masks)) # adding the mask for whe
    masks = np.stack(masks, axis=0)
    # Count the matches and pre-matches
    pre_match_inputs = np.sum(masks, axis=0) > 0
    pre_match = np.sum(pre_match_inputs)
    masks = masks[:, pre_match_inputs]
    br_col = br_col[pre_match_inputs]
    # print(vals)
    # vals = np.array(vals)
    # match is when the br is in the vals of all entries where the mask is true
    match = np.sum(np.all([np.logical_or(np.logical_not(masks[j, :]), np.isin(br_col, vals[j])) for j in range(masks.shape[0])], axis=0))
    accuracy = (match) / pre_match if pre_match > 0 else 1
    support = pre_match/df_test.shape[0]
    # print('pre_match:', pre_match, 'match:', match, 'df_test:', df_test.shape[0])
    if return_vals:
        return support, accuracy
    else:
        print(f'Support for {abr} is : {support}, Confidence: {accuracy}')

# get specs 
def main(filename, mode, add_nn=False):
    df_test_bb = pd.read_csv(f'data/bb_events_pensieve_{mode}.csv')
    df_test_mpc = pd.read_csv(f'data/mpc_events_pensieve_{mode}.csv')
    if add_nn:
        df_test_nn_big = pd.read_csv(f'data/pensieve_events_pensieve_best_big_{mode}.csv')
        df_test_nn_small = pd.read_csv(f'data/pensieve_events_pensieve_best_small_{mode}.csv')
        df_test_nn_mid = pd.read_csv(f'data/pensieve_events_pensieve_best_mid_{mode}.csv')
    print('test set size:', df_test_bb.shape[0])
    with open(filename, 'r') as f:
        spec_file = f.read()
        specs = spec_file.split('-'*50)[:-1]
        specs = [s.strip().split('\n') for s in specs]
        vals = [(s[1].split('output: ')[-1][1:-1].split(',')) for s in specs]
        vals = [[float(v.strip()) for v in val] for val in vals]
        feats = [json.loads(s[0]) for s in specs]
        sup_bb, conf_bb = get_accuracy(df_test_bb, 'bb', feats, vals, return_vals = True)
        sup_mpc, conf_mpc = get_accuracy(df_test_mpc, 'mpc', feats, vals, return_vals = True)   
        if add_nn:
            support_nn_big, conf_nn_big = get_accuracy(df_test_nn_big, 'pensieve', feats, vals, return_vals=True)
            support_nn_small, conf_nn_small = get_accuracy(df_test_nn_small, 'pensieve', feats, vals, return_vals=True)
            support_nn_mid, conf_nn_mid = get_accuracy(df_test_nn_mid, 'pensieve', feats, vals, return_vals=True)
            return round(sup_bb,2), round(conf_bb,2), round(sup_mpc,2), round(conf_mpc,2), round(support_nn_small,2), round(conf_nn_small,2), round(support_nn_mid,2), round(conf_nn_mid,2), round(support_nn_big,2), round(conf_nn_big,2)
    return sup_bb, conf_bb, sup_mpc, conf_mpc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('spec_file', type=str)
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--print_vals', action='store_true')
    parser.add_argument('--add_nn', action='store_true')
    args = parser.parse_args()
    ans = main(args.spec_file, args.mode, args.add_nn)
    if args.print_vals:
        print(ans)