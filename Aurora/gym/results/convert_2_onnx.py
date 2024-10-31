import sys 
sys.path.append('../')
sys.path.append('.')
from model import CustomNetwork_mid, CustomNetwork_big, CustomNetwork_mid_policy_net, CustomNetwork_small_policy_net
import torch

model_name = 'pcc_model_small_10_best'

model = CustomNetwork_small_policy_net()
state_dict = torch.load(model_name+'.pt')

for key in list(state_dict.keys()):
    state_dict[key.replace('mlp_extractor.', '')] = state_dict.pop(key)

state_dict.requires_grad = False
model.load_state_dict(state_dict, strict=False)

model.eval()

dummy_input = torch.randn(1, 30)

torch.onnx.export(model,                     # Model to export
                  dummy_input,               # Dummy input to trace the model
                  f"onnx_models/{model_name}.onnx",              # Output file name
                  export_params=True,        # Store trained parameters (weights) inside the model
                  opset_version=11,          # ONNX version, 11 is commonly used
                  do_constant_folding=True)