# This script is to test whether the specification works

import numpy as np
import torch.onnx
import random
from Models.Aurora import model_benchmark as model

ONNX_DIR = '../onnxs'
MODEL_LIST = ['small', 'mid', 'big']
MODEL = MODEL_LIST[1]
MODEL_TYPES = ['simple', 'parallel', 'concat']
MODEL_TYPE = MODEL_TYPES[0]
NN_MODEL = './results/pcc_model_mid_10_best.pt'

HISTORY = 10

def create_next_step(input):
    next_input = input.copy()
    next_input = np.roll(next_input, -1, axis=1)
    next_input[0][0] = 0.01 + random.random() * 0.01
    next_input[1][0] = 1 + random.random() * 0.01
    return next_input

def get_inputs_array(spec_type, random_seed=0):
    random.seed(random_seed)
    if spec_type==101:
        gene_inputs = np.zeros((3, HISTORY))
        for i in range(HISTORY):
            gene_inputs[0][i] = random.random() * 0.005
            gene_inputs[1][i] = 1 + random.random() * 0.005
            gene_inputs[2][i] = 1
    if spec_type==102:
        gene_inputs = np.zeros((3, HISTORY))
        print("spec 102")
        for i in range(HISTORY):
            gene_inputs[0][i] = 0.5 +random.random() * 0.5
            gene_inputs[1][i] = 5 + random.random()*5
            gene_inputs[2][i] = 2 + random.random()
    if spec_type==2:
        gene_inputs = np.zeros((3, HISTORY))
        for i in range(HISTORY):
            gene_inputs[0][i] = random.random() * 0.01
            gene_inputs[1][i] = 1 + random.random() * 0.01
            gene_inputs[2][i] = 2 + random.random()
    if spec_type==3:
        myinput = get_inputs_array(101)
        myinput2 = myinput.copy()
        myinput2[0][9]=myinput2[0][9]+0.1
        myinput2[1][9] = myinput2[1][9] + 0.1
        gene_inputs = np.concatenate([myinput, myinput2])
    if spec_type==4:
        print("spec type 4")
        myinput = get_inputs_array(102)
        myinput2 = create_next_step(myinput)
        myinput3 = create_next_step(myinput2)
        myinput4 = create_next_step(myinput3)
        myinput5 = create_next_step(myinput4)
        gene_inputs = np.concatenate([myinput, myinput2, myinput3, myinput4, myinput5])
    return(gene_inputs)

def load_model(actor):
    para = torch.load(NN_MODEL, map_location=torch.device('cpu'))
    newpara = {}
    newpara['policy_net.0.weight'] = para["mlp_extractor.policy_net.0.weight"]
    newpara['policy_net.0.bias'] = para["mlp_extractor.policy_net.0.bias"]
    newpara['policy_net.1.weight'] = para["mlp_extractor.policy_net.1.weight"]
    newpara['policy_net.1.bias'] = para["mlp_extractor.policy_net.1.bias"]
    newpara['policy_net.2.weight'] = para["mlp_extractor.policy_net.2.weight"]
    newpara['policy_net.2.bias'] = para["mlp_extractor.policy_net.2.bias"]
    actor.load_state_dict(newpara)
    actor.eval()
    return actor


def main():
    if MODEL_TYPE == 'simple':
        if MODEL == 'mid':
            actor = model.CustomNetwork_mid()
        if MODEL == 'big':
            actor = model.CustomNetwork_mid()
        if MODEL == 'small':
            actor = model.CustomNetwork_mid()

        # load model
        actor = load_model(actor)

        myinput = get_inputs_array(102)
        input = torch.from_numpy(myinput).flatten().to(torch.float32).unsqueeze(0)

        torch_out = actor(input)
        print(torch_out)

    if MODEL_TYPE == 'parallel':
        if MODEL == 'mid':
            actor = model.CustomNetwork_mid_parallel()
        if MODEL == 'big':
            actor = model.CustomNetwork_mid()
        if MODEL == 'small':
            actor = model.CustomNetwork_mid()

        # load model
        actor = load_model(actor)

        input = get_inputs_array(3)
        input = torch.from_numpy(input).flatten().to(torch.float32).unsqueeze(0)
        torch_out = actor(input)
        print(torch_out)

    if MODEL_TYPE == 'concat':
        if MODEL == 'mid':
            actor = model.CustomNetwork_mid_concatnate()
        if MODEL == 'big':
            actor = model.CustomNetwork_mid()
        if MODEL == 'small':
            actor = model.CustomNetwork_mid()

        # load model
        actor = load_model(actor)

        input = get_inputs_array(4)
        input = torch.from_numpy(input).flatten().to(torch.float32).unsqueeze(0)
        torch_out = actor(input)
        print(torch_out)


if __name__ == '__main__':
    main()
