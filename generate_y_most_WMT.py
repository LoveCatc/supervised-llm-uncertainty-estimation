import json
import evaluate
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from joblib import Parallel, delayed
import os
from nltk.translate.bleu_score import sentence_bleu
import argparse

parser = argparse.ArgumentParser(description='Generate the answer for the dataset')
parser.add_argument('--dataset_name', type=str, default='wmt__test', help='the name of the dataset')
parser.add_argument('--model_type', type=str, default='llama_2_7b', help='the type of the model')

args = parser.parse_args()


MOST_ANSWER = 'most_likely_answer'
ANSWER_REF = 'answer_str'
dataset_name = args.dataset_name
model_type = args.model_type
data_json_path = "./test_output/"+dataset_name+"/"+model_type+"/"+dataset_name+"_mextend.json"


data_extend_path = "./test_output/"+dataset_name+"/"+model_type+"/"+dataset_name+"_mextend_bleu.json"
if not os.path.exists(data_extend_path):
    with open(data_json_path) as fr:
        data = json.load(fr)
    data_extend_rouge = deepcopy(data)
else:
    with open(data_extend_path) as fr:
        data_extend_rouge = json.load(fr)

bleu = evaluate.load("bleu")
metric = 'bleu'


def calculate_bleu(d):
    generated_answer = [d[MOST_ANSWER][0].lstrip(),]

    reference = [d[ANSWER_REF].lstrip()]

    if generated_answer[0]=="":
        d[metric]=0
        return 0

    score = bleu.compute(predictions=generated_answer, references=reference)
    d[metric]=score[metric]

    return score


for from_idx in tqdm(range(len(data_extend_rouge))): #len(data_extend_rouge)
    #to_idx = min(from_idx+STEP_SIZE,len(data_extend_rouge))
    #n_job = 8
    #Parallel(n_jobs=n_job, verbose=10)(delayed(calculate_rouge)(data_extend_rouge[i],rouge) for i in range(from_idx,to_idx))

    #if metric not in data_extend_rouge[from_idx]:
    calculate_bleu(data_extend_rouge[from_idx])


    if (from_idx+1)%500==0:
        
        # save the data_extend_rouge
        with open(data_extend_path, 'w') as fw:
            json.dump(data_extend_rouge, fw)
    if from_idx>18000:
        # save the data_extend_rouge
        with open(data_extend_path, 'w') as fw:
            json.dump(data_extend_rouge, fw)
        break



