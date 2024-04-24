from transformers import AutoTokenizer, AutoModelForCausalLM

import numpy as np
#one-layer MLP
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import json

from time import time
from typing import Optional
import os
from generator_cls import INSIDEGenerator
from funs_load_model import load_llama2, load_gemma
from data_entry_new import cnndailymail_formatter,wmt_formatter,triviaqa_formatter,coqa_formatter
import json
from time import time
import argparse

parser = argparse.ArgumentParser(description='Generate the answer for the dataset')
parser.add_argument('--dataset_name', type=str, default='wmt__test', help='the name of the dataset')
parser.add_argument('--model_type', type=str, default='llama_2_7b', help='the type of the model')

args = parser.parse_args()

output_dir = "test_output"
model_type = args.model_type
if model_type=="gemma_7b":
    model_path = "gemma-7b"
    tokenizer_path = "gemma-7b"
elif model_type=="llama_2_7b":
    model_path = "Llama-2-7b-hf-local"
    tokenizer_path = "Llama-2-7b-hf-local"

model,tokenizer = load_llama2(model_path, tokenizer_path)


dataset_name = args.dataset_name
hidden_state_output_dir = output_dir+'/'+dataset_name+'/'+model_type+'/'

PROMPT_TOKENS = 'tokenized_prompt'
Q_BEGIN = 'question_token_start_idx'
Q_END = 'answer_token_start_idx'
QUERY_KEY = 'question_str'
output_token_average_hidden_states = False
len_of_token_hidden_states_output = 0 # if set to zero, then not used
get_query_entropies = False # whether to get the entropy of the output token
get_query_probs = False

    
if model_type == "llama_2_7b":
    layer_list = [16,32]
    num_dim = 4096
elif model_type == "gemma_7b":
    layer_list = [14,28]
    num_dim = 3072


num_entropy_statistics = 4


# generate multiple answers and get the features (statistics of entropy of output logits) of answers
dataset_extend_name = dataset_name+'_extend.json'
dataset_extend_path = hidden_state_output_dir +'/'+dataset_extend_name

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

# if the path not exists, then create the path
if not os.path.exists(hidden_state_output_dir):
    os.makedirs(hidden_state_output_dir)

if os.path.exists(dataset_extend_path):
    with open(dataset_extend_path, "r") as f:
        data_extend = json.load(f)

else:
    time1 = time()
    data_extend = list(data)
    time2 = time()
    print("Time to list the data:", time2-time1)
    
ANSWERS = 'generated_answers'
ANSWER_ENTROPY_STATISTICS = 'answer_entropy_statistics'



if dataset_name.startswith("triviaqa") or dataset_name.startswith("coqa"):
    MAX_LENGTH_OF_GENERATED_SEQUENCE = 30
    eos_words = ['Question:', ' Question:', '\n', '\n\n','\n\n\n','\n\n\n\n', '\n\n\n\n\n','<eos>' ,'Answer:', ' Answer:', 'Q:']
    NUM_GENERATION_PER_PROMPT = 10
    STEP_SIZE=500
elif dataset_name.startswith("cnndaily"):
    MAX_LENGTH_OF_GENERATED_SEQUENCE = 200
    eos_words = ['<end_of_turn>','end_of_turn','<start_of_turn>','start_of_turn']
    NUM_GENERATION_PER_PROMPT = 10
    STEP_SIZE = 20
elif dataset_name.startswith("wmt"):
    MAX_LENGTH_OF_GENERATED_SEQUENCE = 50
    eos_words = ['Q:', '\n', '\n\n','\n\n\n','\n\n\n\n', '\n\n\n\n\n','<eos>' ,'A:','</s><s>']
    NUM_GENERATION_PER_PROMPT = 5
    STEP_SIZE=50
    
TEMPERATURE = 1.0
TOP_P = 1.0



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_entropy_statistics = 4
num_prob_statistics = 6

with torch.no_grad():
    generator = INSIDEGenerator(model=model, tokenizer=tokenizer, layer_list=layer_list, eos_words=eos_words,output_token_average_hidden_states=False,len_of_token_hidden_states_output=0,get_query_entropies=False,get_query_probs=False,layer_dim=num_dim)
    from_idx = 0
    to_idx = from_idx+STEP_SIZE
    find_start_point_flag = False
    
    # skip the processed data
    while not find_start_point_flag:
        for idx,d in enumerate(data_extend[from_idx:to_idx]):
            if (ANSWERS not in d) or len(d[ANSWERS]) == 0:
                find_start_point_flag = True
                break
        if not find_start_point_flag:
            from_idx = to_idx
            to_idx = min(len(data_extend),to_idx+STEP_SIZE)
    
    def load_saved_data(hidden_state_output_dir, from_idx, to_idx, layer_list,len_of_token_hidden_states_output, num_entropy_statistics,num_prob_statistics):
        # Initialize the output
        output_average_tensors = torch.zeros((STEP_SIZE,NUM_GENERATION_PER_PROMPT,len(layer_list), num_dim), dtype=torch.float16)
        output_last_token_tensors = torch.zeros((STEP_SIZE,NUM_GENERATION_PER_PROMPT,len(layer_list), len_of_token_hidden_states_output, num_dim), dtype=torch.float16)
        entropy_output_tensors = torch.zeros((STEP_SIZE,NUM_GENERATION_PER_PROMPT, num_entropy_statistics), dtype=torch.float16)
        prob_output_tensors = torch.zeros((STEP_SIZE,NUM_GENERATION_PER_PROMPT, num_prob_statistics), dtype=torch.float16)

        # check if the hidden states are already generated, if so, load the hidden states
        for idx,layer_idx in enumerate(layer_list):
            if os.path.exists(hidden_state_output_dir+'answer_average_layer_'+str(layer_list[idx])+'_'+str(from_idx)+'_'+str(to_idx)+'.pt'):
                output_average_tensors[:,:,idx,:] = torch.load(hidden_state_output_dir+'answer_average_layer_'+str(layer_idx)+'_'+str(from_idx)+'_'+str(to_idx)+'.pt').to(device)

            if os.path.exists(hidden_state_output_dir+'answer_last_'+str(len_of_token_hidden_states_output)+'_token_layer_'+str(layer_list[0])+'_'+str(from_idx)+'_'+str(to_idx)+'.pt'):
                output_last_token_tensors[:,:,idx,:,:] = torch.load(hidden_state_output_dir+'answer_last_'+str(len_of_token_hidden_states_output)+'_token_layer_'+str(layer_idx)+'_'+str(from_idx)+'_'+str(to_idx)+'.pt').to(device)

            elif os.path.exists(hidden_state_output_dir+'answer_last_5_token_layer_'+str(layer_list[0])+'_'+str(from_idx)+'_'+str(to_idx)+'.pt'):
                print("loading data...", from_idx,to_idx)
                output_last_token_tensors[:,:,idx,:,:] = torch.load(hidden_state_output_dir+'answer_last_5_token_layer_'+str(layer_idx)+'_'+str(from_idx)+'_'+str(to_idx)+'.pt')[:,:,-len_of_token_hidden_states_output:,:].to(device)
        if os.path.exists(hidden_state_output_dir+'answer_entropies_'+str(from_idx)+'_'+str(to_idx)+'.pt'):
            entropy_output_tensors = torch.load(hidden_state_output_dir+'answer_entropies_'+str(from_idx)+'_'+str(to_idx)+'.pt').to(device)
        if os.path.exists(hidden_state_output_dir+'answer_probs_'+str(from_idx)+'_'+str(to_idx)+'.pt'):
            prob_output_tensors = torch.load(hidden_state_output_dir+'answer_probs_'+str(from_idx)+'_'+str(to_idx)+'.pt').to(device)

        return output_average_tensors, output_last_token_tensors, entropy_output_tensors, prob_output_tensors

    # load saved data
    if output_token_average_hidden_states:
        output_average_tensors, output_last_token_tensors, entropy_output_tensors, prob_output_tensors = load_saved_data(hidden_state_output_dir, from_idx, to_idx, layer_list,len_of_token_hidden_states_output, num_entropy_statistics,num_prob_statistics)

    start_idx = from_idx
    for data_i in tqdm(range(start_idx,len(data_extend))):
        d = data_extend[data_i]
        
        # check if this data has been processed before
        if (ANSWERS in d) and len(d[ANSWERS]) > 0:
            if not output_token_average_hidden_states:
                continue
            if torch.sum(torch.abs(output_average_tensors[data_i-from_idx])) > 0 and torch.sum(torch.abs(output_last_token_tensors[data_i-from_idx])) > 0 and torch.sum(torch.abs(entropy_output_tensors[data_i-from_idx])) > 0:
                continue
        


        input_length = d[Q_END]
        data_extend[data_i][ANSWERS] = []
        data_extend[data_i][ANSWER_ENTROPY_STATISTICS] = [[] for _ in range(NUM_GENERATION_PER_PROMPT)]
        prompt_tokens = [d[PROMPT_TOKENS][:d[Q_END]],]

        for i in range(NUM_GENERATION_PER_PROMPT):
            
            if output_token_average_hidden_states:
                sequence, output_average_tensor,output_last_token_tensor,entropy_output_tensor,prob_output_tensor = generator.generate_with_cache(
                    prompt_tokens=prompt_tokens,
                    max_gen_len=MAX_LENGTH_OF_GENERATED_SEQUENCE,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                )
                
                data_extend[data_i][ANSWERS].append(sequence)
                data_extend[data_i][ANSWER_ENTROPY_STATISTICS][i] = entropy_output_tensor.detach().cpu().numpy().tolist()
                output_average_tensors[data_i-from_idx,i] = output_average_tensor
                output_last_token_tensors[data_i-from_idx,i] = output_last_token_tensor
                entropy_output_tensors[data_i-from_idx,i] = entropy_output_tensor
                prob_output_tensors[data_i-from_idx,i] = prob_output_tensor
            else:
                sequence = generator.generate_with_cache(
                    prompt_tokens=prompt_tokens,
                    max_gen_len=MAX_LENGTH_OF_GENERATED_SEQUENCE,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                )
                data_extend[data_i][ANSWERS].append(sequence)
    
        if data_i+1 == to_idx:
            
            
            # save the extended data
            with open(dataset_extend_path, 'w') as f:
                json.dump(data_extend, f)
            
            if output_token_average_hidden_states:
                # save the hidden_states output
                for idx,layer_idx in enumerate(layer_list):
                    torch.save(output_average_tensors[:,:,idx,:], hidden_state_output_dir+'answer_average_layer_'+str(layer_idx)+'_'+str(from_idx)+'_'+str(to_idx)+'.pt')
                    torch.save(output_last_token_tensors[:,:,idx,:,:], hidden_state_output_dir+'answer_last_'+str(len_of_token_hidden_states_output)+'_token_layer_'+str(layer_idx)+'_'+str(from_idx)+'_'+str(to_idx)+'.pt')
                # save the entropy output
                torch.save(entropy_output_tensors, hidden_state_output_dir+'answer_entropies_'+str(from_idx)+'_'+str(to_idx)+'.pt')
                # save the prob output
                torch.save(prob_output_tensors, hidden_state_output_dir+'answer_probs_'+str(from_idx)+'_'+str(to_idx)+'.pt')

            to_idx = min(len(data_extend),to_idx+STEP_SIZE)
            from_idx = data_i+1

            if output_token_average_hidden_states:
                output_average_tensors, output_last_token_tensors, entropy_output_tensors, prob_output_tensors = load_saved_data(hidden_state_output_dir, from_idx, to_idx, layer_list,len_of_token_hidden_states_output, num_entropy_statistics,num_prob_statistics)
            
        if dataset_name=="triviaqa__train" and data_i>20000:
            break
        if dataset_name=="coqa__train" and data_i>18000:
            break
        if dataset_name=="cnndaily__train" and data_i>10000:
            break
        if dataset_name=="wmt__train" and data_i>20000:
            break


