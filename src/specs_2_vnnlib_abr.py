import os
import re

num_features = 48

xmap = {
"X0": "Last8_chunk_bitrate",
"X1": "Last7_chunk_bitrate",
"X2": "Last6_chunk_bitrate",
"X3": "Last5_chunk_bitrate",
"X4": "Last4_chunk_bitrate",
"X5": "Last3_chunk_bitrate",
"X6": "Last2_chunk_bitrate",
"X7": "Last1_chunk_bitrate",
"X8": "Last8_buffer_size",
"X9": "Last7_buffer_size",
"X10": "Last6_buffer_size",
"X11": "Last5_buffer_size",
"X12": "Last4_buffer_size",
"X13": "Last3_buffer_size",
"X14": "Last2_buffer_size",
"X15": "Last1_buffer_size",
"X16": "Last8_throughput",
"X17": "Last7_throughput",
"X18": "Last6_throughput",
"X19": "Last5_throughput",
"X20": "Last4_throughput",
"X21": "Last3_throughput",
"X22": "Last2_throughput",
"X23": "Last1_throughput",
"X24": "Last8_downloadtime",
"X25": "Last7_downloadtime",
"X26": "Last6_downloadtime",
"X27": "Last5_downloadtime",
"X28": "Last4_downloadtime",
"X29": "Last3_downloadtime",
"X30": "Last2_downloadtime",
"X31": "Last1_downloadtime",
"X32": "chunksize1",
"X33": "chunksize2",
"X34": "chunksize3",
"X35": "chunksize4",
"X36": "chunksize5",
"X37": "chunksize6",
"X38": "last2_chunksize1",
"X39": "last1_chunksize1",
"X40": "Last7_chunks_left",
"X41": "Last6_chunks_left",
"X42": "Last5_chunks_left",
"X43": "Last4_chunks_left",
"X44": "Last3_chunks_left",
"X45": "Last2_chunks_left",
"X46": "Last1_chunks_left",
"X47": "Chunks_left"
}

reversed_xmap = {v: k for k, v in xmap.items()}

br_list = [300.0, 750.0, 1200.0, 1850.0, 2850.0, 4300.0]

# responsible for writing the file
def write_vnnlib(X, Y, spec_path):
    with open(spec_path, "w") as f:
        f.write("\n")
        for i in range(len(X)):
            f.write(f"(declare-const X_{i} Real)\n")
        for j in range(6):
            f.write(f"(declare-const Y_{j} Real)\n")
        f.write("\n")
        for i in range(len(X)):
            f.write(f"(assert (>= X_{i} {X[i][0]}))\n")
            if len(X[i]) > 1:
                f.write(f"(assert (<= X_{i} {X[i][1]}))\n")
            
        f.write("\n")
        or_clauses = []
        this_y = [br_list.index(y) for y in Y]
        other_y = [i for i in range(len(br_list)) if i not in this_y]
        for ot in other_y:
            other_y_clauses = [f"(<= Y_{ty} Y_{ot})" for ty in this_y]
            and_stmt = f"(and {' '.join(other_y_clauses)})"
            or_clauses.append(and_stmt)
        f.write(f"(assert (or {' '.join(or_clauses)}))\n")
            
        f.write("\n")
        
name = 'abr_final_full_specs'
spec_file = f'./results/{name}.txt' 
        
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
        Y = [float(y.strip()) for y in Y]
        X = re.findall(spec_pattern, X.strip())
        X = {reversed_xmap[key]: list(map(float, values.split(', '))) for key, values in X}
        for k in xmap.keys():
            if k not in X:
                X[k] = [0.0, 10.0]
        X = {i: X[i] for i in xmap.keys()}
        X = list(X.values())
        if os.path.exists(f'./results/optim_specs/pensieve_vnnlib/{name}') is False:
            os.mkdir(f'./results/optim_specs/pensieve_vnnlib/{name}')
        write_vnnlib(X,Y,f'./results/optim_specs/pensieve_vnnlib/{name}/{idx}.vnnlib')