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
from generator_cls import MMLUGenerator
from funs_load_model import load_llama2
from data_entry_new import mmlu_formatter
import json
from time import time
import argparse

parser = argparse.ArgumentParser(description='Generate the answer for the dataset')
parser.add_argument('--phase', type=str, default='validation', help='the output directory')
parser.add_argument('--model_type', type=str, default='llama_2_7b', help='the type of the model')

args = parser.parse_args()

output_dir = "test_output"
model_type = args.model_type

if model_type.startswith("gemma"):
    os.environ["CUDA_VISIBLE_DEVICES"] = '3, 2'
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

if model_type=="gemma_7b":
    model_path = "gemma-7b"
    tokenizer_path = "gemma-7b"
elif model_type=="llama_2_7b":
    model_path = "Llama-2-7b-hf-local"
    tokenizer_path = "Llama-2-7b-hf-local"
elif model_type == "gemma_2b":
    model_path = "gemma-2b"
    tokenizer_path = "gemma-2b"
elif model_type == "llama_2_13b":
    model_path = "Llama-2-13b-hf-local"
    tokenizer_path = "Llama-2-13b-hf-local"

model,tokenizer = load_llama2(model_path, tokenizer_path)

phase = args.phase

# raise error if phase=="train":
if phase=="train":
    raise ValueError("The phase cannot be train")


hidden_state_output_dir = output_dir+'/MMLU/'+model_type+'/'+phase+'/'

PROMPT_TOKENS = 'tokenized_prompt'
Q_BEGIN = 'question_token_start_idx'
Q_END = 'answer_token_start_idx'
QUERY_KEY = 'question_str'
output_token_average_hidden_states = True
len_of_token_hidden_states_output = 1 # if set to zero, then not used
get_query_entropies = True # whether to get the entropy of the output token
STEP_SIZE = 500

data_tasks = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']

    
if model_type == "llama_2_7b":
    layer_list = [16,32]
    num_dim = 4096
elif model_type == "gemma_7b":
    layer_list = [14,28]
    num_dim = 3072
elif model_type == "gemma_2b":
    layer_list = [9,18]
    num_dim = 2048
elif model_type == "llama_2_13b":
    layer_list = [20,40]
    num_dim = 5120

num_entropy_statistics = 4


data_total = mmlu_formatter(tokenizer=tokenizer, num_example=5,merge_split=False,conv_generation=True)


# if the path not exists, then create the path
if not os.path.exists(hidden_state_output_dir):
    os.makedirs(hidden_state_output_dir)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_entropy_statistics = 4

with torch.no_grad():
    generator = MMLUGenerator(model,tokenizer,layer_list,num_dim)
    
    for task in tqdm(data_tasks):
        dataset_name = 'mmlu__'+task+'__'+phase
        task_output_dir = hidden_state_output_dir+task+'/'
        if not os.path.exists(task_output_dir):
            os.makedirs(task_output_dir)
        if os.path.exists(task_output_dir+str(layer_list[0])+"_output_answer_X.pt"):
            continue
        data = data_total[dataset_name]
        print(len(data))
        num_tokens = 4
        output_answer_X = torch.zeros((len(data),num_tokens,len(layer_list),num_dim))

        data = list(data)
        for i in tqdm(range(0,len(data))):
            d = data[i]
            prompt_tokens = d[PROMPT_TOKENS][:d[Q_END]]
            output_answer_X[i] = generator.generate_single(prompt_tokens)

        # save the result
        
        for idx,layer_idx in enumerate(layer_list):
            torch.save(output_answer_X[:,:,idx,:],task_output_dir+str(layer_idx)+'_output_answer_X.pt')
