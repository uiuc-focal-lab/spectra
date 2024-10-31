import ast
import os
import re
import sys 
import xmap as xm 
import argparse

parser = argparse.ArgumentParser(description='Convert specs to vnnlib format')
parser.add_argument('name', type=str, help='Name of file containing the specs')
args = parser.parse_args()

num_features = 30

reversed_xmap = xm.cc 

xmap = {v: k for k, v in reversed_xmap.items()}

def y_2_neg_symbol(y):
    if y == ['0']:
        return f"(assert (or (and (>= Y_0 {0.0000001})) (and (<= Y_0 {-0.0000001})) ) )\n"
    if y == ['+']:
        return f"(assert (or (and (<= Y_0 {0.0}))))\n"
    if y == ['-']:
        return f"(assert (or (and (>= Y_0 {0.0}))))\n"
    if set(y) == set(['0', '+']):
        return f"(assert (or (and (<= Y_0 {-0.0000001}))))\n"
    if set(y) == set(['0', '-']):
        return f"(assert (or (and (>= Y_0 {0.0000001}))))\n"
    if set(y) == set(['+', '-']):
        return f"(assert (or (and (>= Y_0 {0.0}) (<= Y_0 {0.0}))))\n"

# responsible for writing the file
def write_vnnlib(X, Y, spec_path):
    with open(spec_path, "w") as f:
        f.write("\n")
        for i in range(len(X)):
            f.write(f"(declare-const X_{i} Real)\n")
        f.write(f"(declare-const Y_0 Real)\n")
        f.write("\n")
        for i in range(len(X)):
            f.write(f"(assert (>= X_{i} {X[i][0]}))\n")
            f.write(f"(assert (<= X_{i} {X[i][1]}))\n")
            
        f.write("\n")
        assert_stmt = y_2_neg_symbol(Y)
        f.write(assert_stmt)
            
        f.write("\n")
        
name = args.name
spec_file = f'results/{name}.txt'
        
spec_pattern = r'"(\w+)": \[([-\d.,\s]+)\]'

with open(spec_file, 'r') as f:
    specs = f.read().split('--------------------------------------------------')
    specs = [s.strip() for s in specs]
    specs = [s for s in specs if s != '']
    print('num specs:', len(specs))
    
    for idx in range(len(specs)):
        spec = specs[idx]
        X, Y = spec.split('\n')[0], spec.split('\n')[-1]
        Y = Y.split('output: ')[-1][1:-1].split(', ')
        Y = [y.strip()[1:-1] for y in Y]
        X = re.findall(spec_pattern, X.strip())
        X = {xmap[key]: list(map(float, values.split(', '))) for key, values in X}
        # X_keys = list(X.keys())
        # X_keys.sort()
        X = {i: X[i] for i in range(num_features)}
        X = list(X.values())
        if os.path.exists(f'./results/aurora_vnnlib/{name}') is False:
            os.mkdir(f'./results/aurora_vnnlib/{name}')
        write_vnnlib(X,Y,f'./results/aurora_vnnlib/{name}/{idx}.vnnlib')