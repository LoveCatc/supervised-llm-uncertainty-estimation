import json
import evaluate

import numpy as np
from copy import deepcopy
from tqdm import tqdm
from joblib import Parallel, delayed
import os
# print("before load")
import argparse

parser = argparse.ArgumentParser(description='Generate the answer for the dataset')
parser.add_argument('--dataset_name', type=str, default='coqa__test', help='the name of the dataset')
parser.add_argument('--model_type', type=str, default='llama_2_7b', help='the type of the model')

args = parser.parse_args()

MOST_ANSWER = 'most_likely_answer'
ANSWER_REF = 'answer_str'
dataset_name = args.dataset_name
model_type = args.model_type
data_json_path = "./test_output/"+dataset_name+"/"+model_type+"/"+dataset_name+"_mextend.json"


data_extend_path = "./test_output/"+dataset_name+"/"+model_type+"/"+dataset_name+"_mextend_rouge.json"
if not os.path.exists(data_extend_path):
    with open(data_json_path) as fr:
        data = json.load(fr)
    data_extend_rouge = deepcopy(data)
else:
    with open(data_extend_path) as fr:
        data_extend_rouge = json.load(fr)



rouge_type_list = ['rouge1','rouge2','rougeL','rougeLsum']
rouge_most = ['rouge1_most','rouge2_most','rougeL_most','rougeLsum_most']
threshould = 0.5


rouge = evaluate.load('rouge',keep_in_memory=True)


def calculate_rouge(d,rouge):
    generated_answer = [d[MOST_ANSWER][0][2:].lstrip()]

    reference = [d[ANSWER_REF][2:].lstrip()]


    score = rouge.compute(predictions=generated_answer, references=reference)
    for rouge_idx,rouge_type in enumerate(rouge_type_list):
        
        d[rouge_most[rouge_idx]] = score[rouge_type]


for from_idx in tqdm(range(len(data_extend_rouge))): #len(data_extend_rouge)
    #to_idx = min(from_idx+STEP_SIZE,len(data_extend_rouge))
    #n_job = 8
    #Parallel(n_jobs=n_job, verbose=10)(delayed(calculate_rouge)(data_extend_rouge[i],rouge) for i in range(from_idx,to_idx))
    if 'rouge1_most' not in data_extend_rouge[from_idx]:
        calculate_rouge(data_extend_rouge[from_idx],rouge)


    if (from_idx+1)%500==0:
        
        # save the data_extend_rouge
        with open(data_extend_path, 'w') as fw:
            json.dump(data_extend_rouge, fw)
    if from_idx>18000:
        # save the data_extend_rouge
        with open(data_extend_path, 'w') as fw:
            json.dump(data_extend_rouge, fw)
        break



