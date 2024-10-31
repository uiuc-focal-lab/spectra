from math import floor
import matplotlib.pyplot as plt
import numpy as np
import itertools 
from sklearn.cluster import DBSCAN
import math 
import json
import pandas as pd

def nontrivial_beh(outputs, csrs):
    # find union of all the outputs
    outs = set(outputs[0])
    for i in range(1, len(outputs)):
        outs = outs.union(set(outputs[i]))
        
    return list(outs), (len(outs) < len(csrs))

def spec_fun(ccs, csrs, columns, all_columns, data_files, args, feat_mins, feat_maxs, cc_data, app):
    num_feats = len(columns)    
    columns_str = [str(c) for c in columns]
    parts = args.parts
    rep_thresh = args.rep_thresh
    min_points_thresh = 1/(parts**num_feats)
    coverage = args.coverage
    tight = args.tight 
    mode = 'nontrivial'
    
    # partition input space into regions
    imp_ints = {}
    for cc in ccs:
        ints = {}
        int_csrs = {}
        num_points = 0
        data = cc_data[cc]
        for csr in data.keys():
            points = cc_data[cc][csr]
            num_points += len(points)
            for point in points:
                # determine the region in which this point falls
                idxs = tuple([min(parts-1,floor(parts*(point[i]-feat_mins[i])/(feat_maxs[i]-feat_mins[i]))) for i in range(len(point))])
                if idxs not in ints.keys():
                    ints[idxs] = 0
                    int_csrs[idxs] = set()
                ints[idxs] += 1 
                int_csrs[idxs].add(csr)
        imp_ints[cc] = {}
        print('num points', num_points)
        min_points = max((min_points_thresh*num_points), 1) # atleast 1 point, average number of points in each region
        print('min points', min_points)
        for k in ints.keys():
            if ints[k] >= min_points:
                imp_ints[cc][k] = int_csrs[k] # get the csrs of the region
    print('important intervals identified')
    # identify the behavior
    # get the regions that are common across ccs
    common_ints = set(imp_ints[ccs[0]].keys())
    for cc in ccs[1:]:
        common_ints &= set(imp_ints[cc].keys())
        
    behaviors = {}
    for ci in common_ints:
        outputs = []
        for cc in ccs:
            v = imp_ints[cc][ci]
            outputs.append(v)
        out, check = nontrivial_beh(outputs, csrs)
        if check:
            behaviors[ci] = out
        

    print('mined behaviors', len(behaviors.keys()))
    rep_num_thresh = int(max(10, rep_thresh*len(behaviors.keys()))) # these number of regions should be in each spec, minimum 10
    slack = args.slack # permitted amount of blowup in the spec volume
    eps = int(np.ceil((math.pow(rep_num_thresh, 1/num_feats)-1)/2)) + slack # for tightly bound regions
    print('rep_num_thresh', rep_num_thresh, 'eps', eps)
    coverage = coverage*len(behaviors.keys()) # number of regions to cover
    # conduct optimization

    # all combinations for permissible tightness
    combs = []
    specs = []
    regions_covered = []
    for length in range(tight): #-1, -1,-1): 
        combs.extend(list(itertools.combinations(csrs, length+1)))

    for comb in combs:
        comb = set(comb)
        no_fly_zones = [] # these regions do not have output under this combination :)
        for beh in behaviors.keys():
            if not set(behaviors[beh]).issubset(comb): # or beh in regions_covered:
                no_fly_zones.append(beh)
        # scanline algorithm to merge the remaining regions: we treat the points not in behaviors also as obstacles for now, stop at rectangle with more than representation threshold regions
        comb_beh = [list(beh) for beh in behaviors.keys() if beh not in no_fly_zones]
        if len(comb_beh) == 0:
            continue
        clustering = DBSCAN(eps=eps, metric='chebyshev', min_samples=rep_num_thresh).fit(comb_beh)
        labels = clustering.labels_
        if args.fig:
            if num_feats > 2:
                raise ValueError('cannot plot for more than 2 features')
            colors = ['r', 'g', 'b', 'y', 'c', 'm', 'pink', 'steelblue']
            my_lab = [colors[l] for l in labels]
            plt.scatter([comb_beh[i][0] for i in range(len(comb_beh))], [comb_beh[i][1] for i in range(len(comb_beh))], c=my_lab, s = 1)
            plt.title(f'Combination: {comb}')
            plt.show()
        for l in range(max(labels)+1):
            regions = [comb_beh[i] for i in range(len(comb_beh)) if labels[i] == l]
            # combine the regions to form a spec
            spec_pre = []
            for idx in range(len(comb_beh[0])):
                spec_pre.append([min([r[idx] for r in regions]), max([r[idx] for r in regions])])
            outs = [behaviors[tuple(r)] for r in regions]
            spec_post = set(outs[0])
            for i in range(1, len(outs)):
                spec_post |= set(outs[i])
            if len(spec_post) > tight:
                continue
            specs.append((spec_pre, spec_post))
            regions = [r for r in behaviors.keys() if all(r[c] >= spec_pre[c][0] and r[c] <= spec_pre[c][1] for c in range(len(columns)))]
            regions_covered.extend(regions)
            regions_covered = list(set(regions_covered))
            cov = len(regions_covered)
            if cov >= coverage:
                break
        cov = len(regions_covered)
        if cov >= coverage:
            break
    cov = len(regions_covered)
    print("coverage achieved", cov/len(behaviors.keys()))
    print("number of specs", len(specs))
    specs.sort(key = lambda x: x[0])
    grouped_specs = itertools.groupby(specs, key = lambda x: x[0])
    filtered_specs = []
    for k, g in grouped_specs:
        outputs = []
        for _, spost in g:
            outputs.extend(spost)
        if len(set(outputs)) <= tight:
            filtered_specs.append((k, set(outputs)))
    specs = filtered_specs
    processed_specs = {}
    for id,spec in enumerate(specs):
        pre = spec[0]
        ranges = {}
        for i in range(len(pre)):
            if feat_maxs[i] < feat_mins[i]:
                print(feat_maxs[i], feat_mins[i], pre[i])
                exit(1)
            pre[i][0] = feat_mins[i] + pre[i][0]*(feat_maxs[i]-feat_mins[i])/parts
            pre[i][1] = feat_mins[i] + (pre[i][1]+1)*(feat_maxs[i]-feat_mins[i])/parts
            ranges[columns[i]] = pre[i]

        for co in range(len(all_columns)):
            if co not in ranges.keys():
                ranges[co] = [float('inf'), float('-inf')]
        processed_specs[id] = [ranges, spec[1]]

    if args.filename is not None:
        filename = args.filename
    else:
        columns_str = '_'.join(columns_str)
        filename = f'results/{app}_optim_specs_{mode}_{columns_str}_train.txt'
    print('Writing spec in file:', filename)
    
    with open(filename, 'w') as f:
        
        for file in data_files.values():
            df = pd.read_csv(file)
            # for features other than the considered ones, we will find the min and max based on points that fall within the specs and then write the specs
            counts = {}
            for p in range(df.shape[0]):
                point = df.iloc[p]
                point = point.to_dict()
                for sp in processed_specs.keys():
                    spec = processed_specs[sp]
                    check = True
                    for c in columns:
                        if point[all_columns[c]] < spec[0][c][0] or point[all_columns[c]] > spec[0][c][1]:
                            check = False
                            break
                    if check:
                        if sp not in counts.keys():
                            counts[sp] = 0
                        counts[sp] += 1
                        for co in range(len(all_columns)):
                            if co not in columns:
                                spec[0][co] = [min(spec[0][co][0], point[all_columns[co]]), max(spec[0][co][1], point[all_columns[co]])]
                    processed_specs[sp] = spec
            print('counts', counts)   
                                                        
        for sp in processed_specs.keys():
            spec_dict = {}
            if args.full_spec:
                for c in range(len(all_columns)):
                    spec_dict[all_columns[c]] = [round(r, 2) for r in processed_specs[sp][0][c]]
            else:
                for c in columns:
                    spec_dict[all_columns[c]] = [round(r, 2) for r in processed_specs[sp][0][c]]
            f.write(json.dumps(spec_dict)+'\n'+"output: "+str(processed_specs[sp][1])+'\n\n')
            f.write('-'*50+'\n')