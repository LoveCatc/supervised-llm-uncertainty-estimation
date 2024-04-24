import torch
import numpy as np
import pandas as pd
from collections import Counter
from loguru import logger
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="wmt__test")
parser.add_argument("--model_type", type=str, default="gemma_7b")
args = parser.parse_args()

ENTAILMENT_MODEL_LOCAL = "deberta-large-mnli"
dataset_name = args.dataset_name
output_dir = "./test_output/"+dataset_name+"/"+args.model_type+"/"
GENERATED_QA_LOCAL = output_dir+dataset_name+"_extend.json"
QUESTION_KEY = "question_str"       # string
ANSWERS_KEY = "generated_answers"   # list[list[str]]
SEMANTIC_ENTROPY_KEY = "semantic_entropy"
save_path = output_dir + dataset_name + "_semantic_entropy.json"



entailment_tokenizer = AutoTokenizer.from_pretrained(ENTAILMENT_MODEL_LOCAL)
entailment_model = AutoModelForSequenceClassification.from_pretrained(ENTAILMENT_MODEL_LOCAL).cuda()

with open(GENERATED_QA_LOCAL, "r") as f:
    data_with_answers = json.load(f)

if os.path.exists(save_path):
    with open(save_path, "r") as f:
        data_with_score = json.load(f)
else:
    data_with_score = data_with_answers
    


for ridx in tqdm(range(len(data_with_answers))):
    row = data_with_score[ridx]
    if SEMANTIC_ENTROPY_KEY in row:
        continue
    if ANSWERS_KEY not in row or row[ANSWERS_KEY] == []:
        # check if there is also no answers in data_with_answers
        if ANSWERS_KEY not in data_with_answers[ridx] or data_with_answers[ridx][ANSWERS_KEY] == []:
            continue
        else:
            # check if they are the same question
            if row[QUESTION_KEY] != data_with_answers[ridx][QUESTION_KEY]:
                logger.warning(f"Not the same question in row {ridx}")
                break
            else:
                row[ANSWERS_KEY] = data_with_answers[ridx][ANSWERS_KEY]

    question = row[QUESTION_KEY]

    try:
        answers = sum(row[ANSWERS_KEY], [])     # flatten the list
    except TypeError:
        answers = row[ANSWERS_KEY]

    
    # use only unique answers - follow semantic entropy implementation
    answers_set = list(set(answers))
    num_answers = len(answers_set)

    alist1, alist2, entailment_prompts = [], [], []
    
    # records answer and its semantic cluster - used for semantic entropy
    ans2smt = {answer: i for i, answer in enumerate(answers_set)}

    
    if num_answers == 1:
        row[SEMANTIC_ENTROPY_KEY] = 0
    else:
        for i, ref_answer in enumerate(answers_set):
            for j in range(i+1, len(answers_set)):
                alist1.append(answers_set[i])
                alist2.append(answers_set[j])
                
                qa_1 = question + " " + answers[i]
                qa_2 = question + " " + answers[j]
                
                # not sure, but this seperator is used in semantic uncertainty
                entailment_prompt = qa_1 + "[SEP]" + qa_2
                entailment_prompts.append(entailment_prompt)
                
                # here we just follow semantic uncertainty
                encoded_prompt = entailment_tokenizer.encode(
                    entailment_prompt, padding=True)
                pred = entailment_model(
                    # torch.tensor(
                    #     torch.tensor([encoded_prompt]),
                    #     device="cuda"
                    # )
                    torch.tensor([encoded_prompt], device="cuda")
                )["logits"]
                pred_label = torch.argmax(pred, dim=1)
                
                reversed_prompt = qa_2 + "[SEP]" + qa_1
                encoded_reversed_prompt = entailment_tokenizer.encode(
                    reversed_prompt, padding=True)
                reversed_pred = entailment_model(
                    # torch.tensor(
                    #     torch.tensor([encoded_reversed_prompt]),
                    #     device="cuda"
                    # )
                    torch.tensor([encoded_reversed_prompt], device="cuda")
                )["logits"]
                reversed_pred_label = torch.argmax(reversed_pred, dim=1)
                
                if 0 in pred_label or 0 in reversed_pred_label:
                    pass    # semantically different, do nothing
                else:       # semantically same, merge clusters
                    ans2smt[answers_set[j]] = ans2smt[answers_set[i]]
        
        semantic_group = list(ans2smt.values())
        group_of_answer = [ans2smt[answer] for answer in answers]
        semantic_group_set = set(semantic_group)


        # calculate the number of samples in each cluster
        num_samples_in_cluster = [group_of_answer.count(group_idx) for group_idx in semantic_group_set]

        N = num_answers
        
        semantic_entropy =-1/len(semantic_group_set)*sum([np.log(num_sample/N) for num_sample in num_samples_in_cluster])
        row[SEMANTIC_ENTROPY_KEY] = semantic_entropy


    # save the data
    if (ridx+1) % 500 == 0:
        with open(save_path, "w") as f:
            json.dump(data_with_score, f)


                
                