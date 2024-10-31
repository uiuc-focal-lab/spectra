import pandas as pd 
import json
import numpy as np
import argparse 

def get_accuracy(df_test, abr, feats, vals, return_only_support = False, return_vals = False):
    if len(feats) == 0:
        if return_only_support:
            return 0
        if return_vals:
            return 0, 0
        else:
            print(f'Support for {abr} is: 0, Confidence: 0')
            
    br_col = df_test['change_of_sending_rate'].values
    cols = [df_test[col].values for col in feats[0].keys()]

    # Precompute the condition masks
    masks = []
    for j in range(len(feats)):
        feat_masks = []
        for f, (lower, upper) in feats[j].items():
            feat_masks.append(np.logical_and(cols[list(feats[j].keys()).index(f)] >= lower,
                                              cols[list(feats[j].keys()).index(f)] <= upper))
        masks.append(np.logical_and.reduce(feat_masks))
    masks = np.stack(masks, axis=0)
    # Count the matches and pre-matches
    pre_match = np.sum(np.sum(masks, axis=0) > 0)
    # match is when the br is in the vals of any entry where the mask is true
    donotmatch = np.sum(np.logical_not(np.all([np.logical_or(np.logical_not(masks[j, :]), np.isin(br_col, vals[j])) for j in range(masks.shape[0])], axis=0)))
    accuracy = (pre_match-donotmatch) / pre_match if pre_match > 0 else 0
    support = pre_match/df_test.shape[0]
    if return_only_support:
        return support
    if return_vals:
        return support, accuracy
    else:
        print(f'Support for {abr} is: {support}, Confidence: {accuracy}')

# get specs 
def main(filename, mode, add_nn=False):
    df_test_bbr = pd.read_csv(f'data/bbr_events_{mode}_filtered.csv') 
    df_test_cubic = pd.read_csv(f'data/cubic_events_{mode}_filtered.csv')
    if add_nn:
        df_test_nn_small = pd.read_csv(f'data/aurora_df_small_{mode}_filtered.csv')
        change_rate = list(df_test_nn_small['actions'].values)
        change_rate = [float(c.split('tensor([',1)[-1].split('])')[0]) for c in change_rate]
        df_test_nn_small['change_of_sending_rate'] = [np.where(c > 0, '+', np.where(c < 0, '-', '0')) for c in change_rate]
        df_test_nn_mid = pd.read_csv(f'data/aurora_df_mid_{mode}_filtered.csv')
        change_rate = list(df_test_nn_mid['actions'].values)
        change_rate = [float(c.split('tensor([',1)[-1].split('])')[0]) for c in change_rate]
        df_test_nn_mid['change_of_sending_rate'] = [np.where(c > 0, '+', np.where(c < 0, '-', '0')) for c in change_rate]
        df_test_nn_mid_original = pd.read_csv(f'data/aurora_df_mid_original_{mode}_filtered.csv')
        change_rate = list(df_test_nn_mid_original['actions'].values)
        change_rate = [float(c.split('tensor([',1)[-1].split('])')[0]) for c in change_rate]
        df_test_nn_mid_original['change_of_sending_rate'] = [np.where(c > 0, '+', np.where(c < 0, '-', '0')) for c in change_rate]
    print('test set size:', df_test_bbr.shape[0])
    with open(filename, 'r') as f: # cc_optim_specs_nontrivial_0_1_2_8_9_10_16_17_18_train
        spec_file = f.read()
        specs = spec_file.split('-'*50)[:-1]
        specs = [s.strip().split('\n') for s in specs]
        vals = [(s[1].split('output: ')[-1][1:-1].split(',')) for s in specs]
        vals = [[v.strip()[1:-1] for v in val] for val in vals]
        feats = [json.loads(s[0]) for s in specs]
        support_bbr, conf_bbr = get_accuracy(df_test_bbr, 'bbr', feats, vals, return_vals = True)
        support_cubic, conf_cubic = get_accuracy(df_test_cubic, 'cubic', feats, vals, return_vals=True)   
        if add_nn:
            support_nn_small, conf_nn_small = get_accuracy(df_test_nn_small, 'aurora', feats, vals, return_vals=True)
            support_nn_mid, conf_nn_mid = get_accuracy(df_test_nn_mid, 'aurora', feats, vals, return_vals=True)
            support_nn_mid_original, conf_nn_mid_original = get_accuracy(df_test_nn_mid_original, 'aurora', feats, vals, return_vals=True)
            return round(support_bbr,2), round(conf_bbr,2), round(support_cubic,2), round(conf_cubic,2), round(support_nn_small,2), round(conf_nn_small,2), round(support_nn_mid,2), round(conf_nn_mid,2), round(support_nn_mid_original,2), round(conf_nn_mid_original,2)
        return support_bbr, conf_bbr, support_cubic, conf_cubic
    
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