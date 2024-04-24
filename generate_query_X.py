from transformers import AutoTokenizer, AutoModelForCausalLM

import numpy as np
#one-layer MLP
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import json
import os

from time import time
from typing import Optional
import os
from torch.nn import functional as F
from tqdm import tqdm
from funs_load_model import load_llama2
from data_entry_new import cnndailymail_formatter,wmt_formatter,triviaqa_formatter,coqa_formatter
import json
from funs_get_feature_X import get_average_hidden_states, get_last_token_hidden_states, get_entropy_statistics,get_prob_statistics
import argparse

parser = argparse.ArgumentParser(description='Generate the answer for the dataset')
parser.add_argument('--dataset_name', type=str, default='wmt__test', help='the name of the dataset')
parser.add_argument('--model_type', type=str, default='llama_2_7b', help='the type of the model')

args = parser.parse_args()

if args.model_type.startswith("gemma"):
    os.environ["CUDA_VISIBLE_DEVICES"] = '3, 2, 0'
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

output_dir = "test_output"
model_type = args.model_type
if model_type=="gemma_7b":
    model_path = "gemma-7b"
    tokenizer_path = "gemma-7b"
elif model_type=="gemma_2b":
    model_path = "gemma-2b"
    tokenizer_path = "gemma-2b"
elif model_type=="llama_2_7b":
    model_path = "Llama-2-7b-hf-local"
    tokenizer_path = "Llama-2-7b-hf-local"
elif model_type=="llama_2_13b":
    model_path = "Llama-2-13b-hf-local"
    tokenizer_path = "Llama-2-13b-hf-local"

model,tokenizer = load_llama2(model_path, tokenizer_path)


dataset_name = args.dataset_name
hidden_state_output_dir = output_dir+'/'+dataset_name+'/'+model_type+'/'

PROMPT_TOKENS = 'tokenized_prompt'
Q_BEGIN = 'question_token_start_idx'
Q_END = 'answer_token_start_idx'
if dataset_name.startswith("triviaqa"):
    data = triviaqa_formatter(tokenizer=tokenizer,num_example=3,cache=True)
    data = data[dataset_name]
elif dataset_name.startswith("coqa"):
    data = coqa_formatter(tokenizer=tokenizer, num_example=3, cache=True)
    if dataset_name.endswith("test"):
        data = data["test"]
    elif dataset_name.endswith("train"):
        data = data["train"]
elif dataset_name.startswith("cnndaily"):
    time1 = time()
    data = cnndailymail_formatter(tokenizer=tokenizer, num_example=3, cache=True)
    time2 = time()
    print("Time to load the data:", time2-time1)
elif dataset_name.startswith("wmt"):
    data = wmt_formatter(tokenizer=tokenizer, num_example=3, cache=True,conv_generation=True)


    data = data[dataset_name]
if dataset_name.endswith("test"):
    # truncate data to 20000
    data = list(data.select(range(min(2000,data.num_rows))))
elif dataset_name.startswith("wmt"):
    data = list(data.select(range(min(20000,data.num_rows))))  





output_token_average_hidden_states = True
len_of_token_hidden_states_output = 1 # if set to zero, then not used
get_query_entropies = True # whether to get the entropy of the output token
get_query_probs = True
num_queries = len(data)

print("queries to be processed: ", num_queries)
    
if model_type == "llama_2_7b":
    layer_list = [16,32]
    num_dim = 4096
elif model_type == "llama_2_13b":
    layer_list = [20,40]
    num_dim = 5120
elif model_type == "gemma_7b":
    layer_list = [14,28]
    num_dim = 3072
elif model_type == "gemma_2b":
    layer_list = [9,18]
    num_dim = 2048



num_entropy_statistics = 4

# initialize output_tensor as num_layers x num_queries x num_dim
if output_token_average_hidden_states:
    output_average_tensor = torch.zeros(( num_queries,len(layer_list), num_dim), dtype=torch.float16)
if len_of_token_hidden_states_output > 0:
    output_last_token_tensor = torch.zeros((num_queries,len(layer_list), len_of_token_hidden_states_output, num_dim), dtype=torch.float16)
if get_query_entropies:
    entropy_output_tensor = torch.zeros((num_queries, num_entropy_statistics), dtype=torch.float16)
if get_query_probs:
    prob_output_tensor = torch.zeros((num_queries,6),dtype=torch.float16)


device = "cuda" if torch.cuda.is_available() else "cpu"
# forward and get features of the query
for data_i, d in tqdm(enumerate(data)):

    q_begin = d[Q_BEGIN]
    q_end = d[Q_END]
    

    prompt_token = d[PROMPT_TOKENS][:q_end]


    # convert prompt_token to tensor
    prompt_token = torch.tensor(prompt_token).unsqueeze(0)
    prompt_token = prompt_token.to(device)

    
    outputs = model.forward(prompt_token, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    

    
    if not os.path.exists(hidden_state_output_dir):
        os.makedirs(hidden_state_output_dir)

    
    if output_token_average_hidden_states:
        output_average_tensor[data_i] = get_average_hidden_states(hidden_states,layer_list, q_begin, q_end, num_dim=num_dim)
    if len_of_token_hidden_states_output > 0:
        output_last_token_tensor[data_i] = get_last_token_hidden_states(hidden_states,layer_list, q_end, len_of_token_hidden_states_output,num_dim=num_dim)

    if get_query_entropies:
        entropy_output_tensor[data_i,:] = get_entropy_statistics(outputs.logits,q_begin,q_end)

    if get_query_probs:
        prob_output_tensor[data_i,:] = get_prob_statistics(outputs.logits,prompt_token,q_begin,q_end)
            
# save the hidden_states output
for idx,layer_idx in enumerate(layer_list):
    if output_token_average_hidden_states:
        torch.save(output_average_tensor[:,idx,:], hidden_state_output_dir+'query_average_layer_'+str(layer_idx)+'.pt')
    if len_of_token_hidden_states_output > 0:
        torch.save(output_last_token_tensor[:,idx,:,:], hidden_state_output_dir+'query_last_'+str(len_of_token_hidden_states_output)+'_token_layer_'+str(layer_idx)+'.pt')

# release the memory
if output_token_average_hidden_states:
    del output_average_tensor
if len_of_token_hidden_states_output > 0:
    del output_last_token_tensor

# save the entropy output
if get_query_entropies:
    torch.save(entropy_output_tensor, hidden_state_output_dir+'query_entropies.pt')
    # release the memory
    del entropy_output_tensor

# save the prob output
if get_query_probs:
    torch.save(prob_output_tensor, hidden_state_output_dir+'query_probs.pt')
    # release the memory
    del prob_output_tensor