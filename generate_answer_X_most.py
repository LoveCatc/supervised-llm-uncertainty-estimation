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
from funs_load_model import load_llama2
from data_entry_new import wmt_formatter, coqa_formatter,triviaqa_formatter
import json
import argparse

parser = argparse.ArgumentParser(description='Generate the answer for the dataset')
parser.add_argument('--dataset_name', type=str, default='coqa__test', help='the name of the dataset')
parser.add_argument('--model_type', type=str, default='llama_2_7b', help='the type of the model')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2, 3'

output_dir = "test_output"
model_type = args.model_type
print(model_type)
if model_type=="llama_2_7b":
    model_path = "Llama-2-7b-hf-local"
    tokenizer_path = "Llama-2-7b-hf-local"
elif model_type=="gemma_7b":
    model_path = "gemma-7b"
    tokenizer_path = "gemma-7b"

dataset_name = args.dataset_name
print(dataset_name)

model,tokenizer = load_llama2(model_path, tokenizer_path)





hidden_state_output_dir = output_dir+'/'+dataset_name+'/'+model_type+'/'

PROMPT_TOKENS = 'tokenized_prompt'
Q_BEGIN = 'question_token_start_idx'
Q_END = 'answer_token_start_idx'
output_token_average_hidden_states = True
len_of_token_hidden_states_output = 1 # if set to zero, then not used
get_query_entropies = True # whether to get the entropy of the output token


    
if model_type == "llama_2_7b":
    layer_list = [16,32]
    num_dim = 4096
elif model_type == "gemma_7b":
    layer_list = [14,28]
    num_dim = 3072


num_entropy_statistics = 4
num_prob_statistics = 6


# generate multiple answers and get the features (statistics of entropy of output logits) of answers
dataset_extend_name = dataset_name+'_mextend.json'
dataset_extend_path = hidden_state_output_dir +'/'+dataset_extend_name
# if the path not exists, then create the path
if not os.path.exists(dataset_extend_path):
    if not os.path.exists(hidden_state_output_dir):
        os.makedirs(hidden_state_output_dir)

    if dataset_name.startswith("wmt"):
        data = wmt_formatter(tokenizer=tokenizer, num_example=3, cache=True,conv_generation=True)
        data = data[dataset_name]

        
    elif dataset_name.startswith("coqa"):
        data = coqa_formatter(tokenizer=tokenizer,num_example=3,cache=True)
        if dataset_name.endswith("train"):
            data = data["train"]
        elif dataset_name.endswith("test"):
            data = data["test"]

    elif dataset_name.startswith("triviaqa"):
        data = triviaqa_formatter(tokenizer=tokenizer,num_example=3,cache=True)
        data = data[dataset_name]  

    
    if dataset_name.endswith("train"):
        data = data.select(range(min(20000,data.num_rows)))
        print(data.num_rows)
    elif dataset_name.endswith("test"):
        data = data.select(range(min(2000,data.num_rows)))
    

     

    data_extend = list(data)
    num_query = len(data_extend)

    # Initialize the output
    output_average_tensors = torch.zeros((num_query,len(layer_list), num_dim), dtype=torch.float16)
    output_last_token_tensors = torch.zeros((num_query,len(layer_list), len_of_token_hidden_states_output, num_dim), dtype=torch.float16)
    entropy_output_tensors = torch.zeros((num_query,num_entropy_statistics), dtype=torch.float16)
    prob_output_tensors = torch.zeros((num_query,num_prob_statistics), dtype=torch.float16)
    
else:
    with open(dataset_extend_path) as fr:
        data_extend = json.load(fr)
    num_query = len(data_extend)
    # Initialize the output
    output_average_tensors = torch.zeros((num_query,len(layer_list), num_dim), dtype=torch.float16)
    output_last_token_tensors = torch.zeros((num_query,len(layer_list), len_of_token_hidden_states_output, num_dim), dtype=torch.float16)
    
    # load the saved result:
    for idx,layer_idx in enumerate(layer_list):
        output_average_tensors[:,idx,:] = torch.load(hidden_state_output_dir+'answerm_average_layer_'+str(layer_idx)+'.pt')
        output_last_token_tensors[:,idx,:,:] = torch.load(hidden_state_output_dir+'answerm_last_'+str(len_of_token_hidden_states_output)+'_token_layer_'+str(layer_idx)+'.pt').reshape(num_query,len_of_token_hidden_states_output,num_dim)
    entropy_output_tensors = torch.load(hidden_state_output_dir+'answerm_entropies.pt')
    if os.path.exists(hidden_state_output_dir+'answerm_probs.pt'):
        prob_output_tensors = torch.load(hidden_state_output_dir+'answerm_probs.pt')
    else:
        prob_output_tensors = torch.zeros((num_query,num_prob_statistics), dtype=torch.float16)
    



print("queries to be processed: ", len(data_extend))


MOST_ANSWER = 'most_likely_answer'
MOST_ANSWER_ENTROPY_STATISTICS = 'most_likely_answer_entropy_statistics'


if dataset_name.startswith("triviaqa") or dataset_name.startswith("coqa"):
    MAX_LENGTH_OF_GENERATED_SEQUENCE = 30
    eos_words = ['Question:', ' Question:', '\n', '\n\n','\n\n\n','\n\n\n\n', '\n\n\n\n\n','<eos>' ,'Answer:', ' Answer:', 'Q:']
    STEP_SIZE=50
elif dataset_name.startswith("cnndaily"):
    MAX_LENGTH_OF_GENERATED_SEQUENCE = 200
    eos_words = ['<end_of_turn>','end_of_turn','<start_of_turn>','start_of_turn']
    STEP_SIZE=20
elif dataset_name.startswith("wmt"):
    MAX_LENGTH_OF_GENERATED_SEQUENCE = 100
    eos_words = ['Q:', ' Q:', '\n', '\n\n','\n\n\n','\n\n\n\n', '\n\n\n\n\n','<eos>' ,'A:', ' A:','</s><s>']
    STEP_SIZE=50

TOP_P = 1.0
period_token_id = tokenizer('. ')['input_ids'][1]

question_framing_ids = [[tokenizer(eos_token)['input_ids'][1]] for eos_token in eos_words]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



with torch.no_grad():
    generator = INSIDEGenerator(model=model, tokenizer=tokenizer, layer_list=layer_list, len_of_token_hidden_states_output=len_of_token_hidden_states_output,eos_words=eos_words,layer_dim=num_dim)


    # save_flag = False

    for data_i,d in tqdm(enumerate(data_extend)):

        
        if MOST_ANSWER in d and len(d[MOST_ANSWER]) > 0:
            if torch.sum(torch.abs(output_average_tensors[data_i])) > 0 and torch.sum(torch.abs(output_last_token_tensors[data_i])) > 0 and torch.sum(torch.abs(entropy_output_tensors[data_i])) > 0 and torch.sum(torch.abs(prob_output_tensors[data_i])) > 0:
                continue
        
        
        
        input_length = d[Q_END]
        prompt_tokens = [d[PROMPT_TOKENS][:input_length],]
        data_extend[data_i][MOST_ANSWER] = []
        data_extend[data_i][MOST_ANSWER_ENTROPY_STATISTICS] = []

        """
        token_tensor = torch.tensor(prompt_tokens).to(device)
        print(tokenizer.decode(token_tensor[0]))
        #print(d['answer_str'])
        token_answer = [d[PROMPT_TOKENS][d[Q_BEGIN]:d[Q_END]],]
        token_answer_tensor = torch.tensor(token_answer).to(device)
        print(tokenizer.decode(token_answer_tensor[0]))
        """

        # print(tokenizer.decode(torch.tensor(prompt_tokens[0])))
        sequence, output_average_tensor,output_last_token_tensor,entropy_output_tensor,prob_output_tensor = generator.generate_with_cache(
                prompt_tokens=prompt_tokens,
                max_gen_len=MAX_LENGTH_OF_GENERATED_SEQUENCE,
                temperature=-1,
                top_p=TOP_P,
        )
        
        
        data_extend[data_i][MOST_ANSWER] = sequence
        data_extend[data_i][MOST_ANSWER_ENTROPY_STATISTICS] = entropy_output_tensor.detach().cpu().numpy().tolist()
        output_average_tensors[data_i] = output_average_tensor
        output_last_token_tensors[data_i] = output_last_token_tensor
        entropy_output_tensors[data_i] = entropy_output_tensor
        prob_output_tensors[data_i] = prob_output_tensor

        
        if (data_i+1)%STEP_SIZE == 0 or data_i+1 == num_query:
            # save the extended data with most_likely_answer
            with open(dataset_extend_path, 'w') as f:
                json.dump(data_extend, f)



            # save the hidden_states output
            for idx,layer_idx in enumerate(layer_list):
                torch.save(output_average_tensors[:,idx,:], hidden_state_output_dir+'answerm_average_layer_'+str(layer_idx)+'.pt')
                torch.save(output_last_token_tensors[:,idx,:,:], hidden_state_output_dir+'answerm_last_'+str(len_of_token_hidden_states_output)+'_token_layer_'+str(layer_idx)+'.pt')

            # save the entropy output
            torch.save(entropy_output_tensors, hidden_state_output_dir+'answerm_entropies.pt')
            torch.save(prob_output_tensors, hidden_state_output_dir+'answerm_probs.pt')
        
