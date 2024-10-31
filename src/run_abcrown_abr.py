import os
import sys
import ast
from torch import tensor
import torch
import argparse

os.environ['MKL_THREADING_LAYER'] = 'GNU'

parser = argparse.ArgumentParser(description='Run ABCROWN on Pensieve model')
parser.add_argument('model_name', type=str)
parser.add_argument('--attack', action='store_true')
args = parser.parse_args()
# create yaml
vnn_dir_path = 'specs/abr_final_full_specs/'
model_type = args.model_name
onnx_model = f'models/pensieve_onnx_models/{model_type}.onnx'
yaml_path = f'yaml/abr_final_full_specs/'
running_result_path = 'pensieve_abcrown_running_result/'

if args.attack:
    vnn_dir_path += 'attack_specs/'
    yaml_path += 'attack/'
    running_result_path = 'pensieve_abcrown_running_result/attack/'

if not os.path.exists(running_result_path):
    os.makedirs(running_result_path)
if not os.path.exists(yaml_path):
    os.makedirs(yaml_path)

def create_yaml(yaml, vnn_path, onnx_path, inputshape=6):
    with open(yaml, mode='w') as f:
        f.write("general:\n  enable_incomplete_verification: False\n  conv_mode: matrix\n")
        f.write(f'model:\n  onnx_path: {onnx_path}\n')
        f.write('  onnx_quirks: \"{\'Reshape\': {\'fix_batch_size\': True}}\"\n')
        f.write(f'  input_shape: [-1, {inputshape}, 8]\n')
        f.write(f'specification:\n  vnnlib_path: {vnn_path}\n')
        # f.write(f'attack:\n  pgd_steps: 1000\n') #   pgd_restarts: 1000\n
        f.write(f'bab:\n  timeout: 600\n')
        if args.attack:
            f.write(f'attack:\n  pgd_steps: 1000\n  pgd_restarts: 1000\n')
        # f.write("solver:\n  batch_size: 1\nbab:\n  branching:\n    method: kfsb") #     method: sb\n


def main(abcrown_path):
    for i in range(len(os.listdir(vnn_dir_path))):
        vnn_path = vnn_dir_path + f'{i}.vnnlib'
        # print('VNN Path:', vnn_path)
        # exit()
        if not os.path.exists(vnn_path):
            continue
        print('VNN Path:', vnn_path)
        onnx_path = onnx_model
        yaml = yaml_path + f'/{model_type}_{i}.yaml'
        create_yaml(yaml, vnn_path, onnx_path)
        cmd = f"python {abcrown_path} --config {yaml} | tee {running_result_path}/{model_type}_{i}.txt"
        print('running command:', cmd)
        os.system(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify abr with abcrown.")
    parser.add_argument('--abcrown', type=str, help="Path to the abcrown verifier.", default='complete_verifier/abcrown.py')
    args = parser.parse_args()
    main(args.abcrown_path)

