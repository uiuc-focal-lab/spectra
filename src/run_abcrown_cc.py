import os
import sys
import argparse

os.environ['MKL_THREADING_LAYER'] = 'GNU'

parser = argparse.ArgumentParser(description='Run ABCROWN on Aurora model')
parser.add_argument('model_type', type=str)
args = parser.parse_args()
model_type = args.model_type
# create yaml
vnn_dir_path = 'specs/cc_final_full_specs/'
onnx_model = f'models/aurora_onnx_models/aurora_{model_type}.onnx'
yaml_path = 'yaml/cc_final_full_specs/'
running_result_path = 'aurora_abcrown_running_result/'

if not os.path.exists(running_result_path):
    os.makedirs(running_result_path)
if not os.path.exists(yaml_path):
    os.makedirs(yaml_path)

def create_yaml(yaml, vnn_path, onnx_path, inputshape=6):
    with open(yaml, mode='w') as f:
        f.write("general:\n  enable_incomplete_verification: False\n  conv_mode: matrix\n")
        f.write(f'model:\n  onnx_path: {onnx_path}\n')
        f.write(f'specification:\n  vnnlib_path: {vnn_path}\n')
        f.write("solver:\n  batch_size: 2048\nbab:\n  timeout: 600\n  branching:\n    method: sb\n    input_split:\n      enable: True") # 


def main(abcrown_path):
    for i in range(len(os.listdir(vnn_dir_path))):
        vnn_path = vnn_dir_path + f'{i}.vnnlib'
        if not os.path.exists(vnn_path):
            continue
        onnx_path = onnx_model
        yaml = yaml_path + f'/aurora_{model_type}_{i}.yaml'
        create_yaml(yaml, vnn_path, onnx_path)
        cmd = f"python {abcrown_path} --config {yaml} | tee {running_result_path}/aurora_model.txt"
        os.system(cmd)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Verify cc  with abcrown.")
    parser.add_argument('--abcrown_path', type=str, help="Path to the abcrown verifier.", default='complete_verifier/abcrown.py')
    args = parser.parse_args()
    main(args.abcrown_path)