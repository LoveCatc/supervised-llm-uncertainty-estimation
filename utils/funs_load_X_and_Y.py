import json
from pathlib import Path
from typing import *

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from data_entry_new import mmlu_formatter, webgpt_formatter


def load_MMLU_X_Y(phase,model_name,with_entropy=True,MMLU_TASKS='all'):
    
    task_list = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']

    if MMLU_TASKS=='Group1':
        task_list = task_list[:40]
    elif MMLU_TASKS=='Group2':
        task_list = task_list[40:]
    elif MMLU_TASKS=='all':
        pass
    else:
        raise ValueError("MMLU_TASKS should be 'Group1','Group2' or 'all'")

    result_dir = 'test_output/MMLU/'+model_name+'/'+phase+'/'
    if model_name=="gemma_7b":
        other1_model_name = "llama_2_7b"
        other1_layer_list=[16,32]
        other1_name = "other-7B-"
        other2_model_name = "llama_2_13b"
        other2_layer_list=[20,40]
        other2_name = "other-13B-"
    elif model_name=="llama_2_7b":
        other1_model_name = "gemma_7b"
        other1_layer_list=[14,28]
        other1_name = "other-7B-"
        other2_model_name = "gemma_2b"
        other2_layer_list=[9,18]
        other2_name = "other-2B-"

    other1_result_dir = 'test_output/MMLU/'+other1_model_name+'/'+phase+'/'
    other2_result_dir = 'test_output/MMLU/'+other2_model_name+'/'+phase+'/'


    if model_name=="gemma_7b":
        layer_list=[14,28]
        tokenizer = AutoTokenizer.from_pretrained("gemma-7b")
        data_total = mmlu_formatter(tokenizer=tokenizer, num_example=5,merge_split=False,conv_generation=True)
    elif model_name=="llama_2_7b":
        layer_list=[16,32]
        tokenizer = AutoTokenizer.from_pretrained("Llama-2-7b-hf-local")
        data_total = mmlu_formatter(tokenizer=tokenizer, num_example=5,merge_split=False,conv_generation=True)   
    

    
    for task_idx,task in enumerate(task_list):
        target_dir = result_dir+task+'/'
        other1_target_dir = other1_result_dir+task+'/'
        other2_target_dir = other2_result_dir+task+'/'

        logits = torch.load(target_dir+'query_logits.pt')
        other1_logits = torch.load(other1_target_dir+'query_logits.pt')
        other2_logits = torch.load(other2_target_dir+'query_logits.pt')

        argmax_idx = torch.argmax(logits,dim=1)
        answer_strs = data_total['mmlu__'+task+'__'+phase]['answer_str']


        # map 'A' 'B' 'C' 'D' to 0 1 2 3
        answer_idx = [0 if answer_strs[i]=='A' else 1 if answer_strs[i]=='B' else 2 if answer_strs[i]=='C' else 3 for i in range(len(answer_strs))]
        answer_idx = torch.tensor(answer_idx)
        Y_new = (answer_idx==argmax_idx)

        def get_file_name_list(layer_list):
            query_average_mid_layer_name = 'query_average_layer_'+str(layer_list[0])+'.pt'
            query_average_last_layer_name = 'query_average_layer_'+str(layer_list[1])+'.pt'
            query_last1_token_mid_layer_name = 'query_last_1_token_layer_'+str(layer_list[0])+'.pt'
            query_last1_token_last_layer_name = 'query_last_1_token_layer_'+str(layer_list[1])+'.pt'
            answerm_mid_layer_name = str(layer_list[0])+'_output_answer_X.pt'
            answerm_last_layer_name = str(layer_list[1])+'_output_answer_X.pt'
            return query_average_mid_layer_name,query_average_last_layer_name,query_last1_token_mid_layer_name,query_last1_token_last_layer_name,answerm_mid_layer_name,answerm_last_layer_name

        file_name_list = get_file_name_list(layer_list)
        other1_file_name_list = get_file_name_list(other1_layer_list)
        other2_file_name_list = get_file_name_list(other2_layer_list)

        def get_new_X(dir,file_names,argmax_idx):
            data_list = []
            for file_name in file_names:
                data = torch.load(dir+file_name)
                if file_name.startswith('query_last_1_token'):
                    data = data.squeeze()
                    if len(data.shape)==1:
                        data = data.unsqueeze(0)
                elif file_name.endswith('answer_X.pt'):
                    data_new = torch.stack([data[i,argmax_idx[i],:] for i in range(len(argmax_idx))])
                    data = data_new

                data_list.append(data)

            return data_list
    
        query_average_mid_layer_new,query_average_last_layer_new,query_last1_token_mid_layer_new,query_last1_token_last_layer_new,answerm_mid_layer_new,answerm_last_layer_new = get_new_X(target_dir,file_name_list,argmax_idx)
        other1_query_average_mid_layer_new,other1_query_average_last_layer_new,other1_query_last1_token_mid_layer_new,other1_query_last1_token_last_layer_new,other1_answerm_mid_layer_new,other1_answerm_last_layer_new = get_new_X(other1_target_dir,other1_file_name_list,argmax_idx)
        other2_query_average_mid_layer_new,other2_query_average_last_layer_new,other2_query_last1_token_mid_layer_new,other2_query_last1_token_last_layer_new,other2_answerm_mid_layer_new,other2_answerm_last_layer_new = get_new_X(other2_target_dir,other2_file_name_list,argmax_idx)



        # get the corresponding argmax_idx along dim=1

        query_answer_ave_mid_new = torch.cat((query_average_mid_layer_new,answerm_mid_layer_new),dim=1)
        query_answer_ave_last_new = torch.cat((query_average_last_layer_new,answerm_last_layer_new),dim=1)
        query_answer_last_token_mid_new = torch.cat((query_last1_token_mid_layer_new,answerm_mid_layer_new),dim=1)
        query_answer_last_token_last_new = torch.cat((query_last1_token_last_layer_new,answerm_last_layer_new),dim=1)

        other1_query_answer_ave_mid_new = torch.cat((other1_query_average_mid_layer_new,other1_answerm_mid_layer_new),dim=1)
        other1_query_answer_ave_last_new = torch.cat((other1_query_average_last_layer_new,other1_answerm_last_layer_new),dim=1)
        other1_query_answer_last_token_mid_new = torch.cat((other1_query_last1_token_mid_layer_new,other1_answerm_mid_layer_new),dim=1)
        other1_query_answer_last_token_last_new = torch.cat((other1_query_last1_token_last_layer_new,other1_answerm_last_layer_new),dim=1)

        other2_query_answer_ave_mid_new = torch.cat((other2_query_average_mid_layer_new,other2_answerm_mid_layer_new),dim=1)
        other2_query_answer_ave_last_new = torch.cat((other2_query_average_last_layer_new,other2_answerm_last_layer_new),dim=1)
        other2_query_answer_last_token_mid_new = torch.cat((other2_query_last1_token_mid_layer_new,other2_answerm_mid_layer_new),dim=1)
        other2_query_answer_last_token_last_new = torch.cat((other2_query_last1_token_last_layer_new,other2_answerm_last_layer_new),dim=1)

        probs_new = torch.nn.Softmax(dim=1)(logits)
        other1_probs_new = torch.nn.Softmax(dim=1)(other1_logits)
        other2_probs_new = torch.nn.Softmax(dim=1)(other2_logits)

        entropy_new = -torch.sum(probs_new*torch.log(probs_new),dim=1).reshape(-1,1)
        other1_entropy_new = -torch.sum(other1_probs_new*torch.log(other1_probs_new),dim=1).reshape(-1,1)
        other2_entropy_new = -torch.sum(other2_probs_new*torch.log(other2_probs_new),dim=1).reshape(-1,1)

        sorted_probs = torch.sort(probs_new,dim=1,descending=True).values
        other1_sorted_probs = torch.sort(other1_probs_new,dim=1,descending=True).values
        other2_sorted_probs = torch.sort(other2_probs_new,dim=1,descending=True).values

        entropy_features = torch.cat((entropy_new,sorted_probs),dim=1)
        other1_entropy_features = torch.cat((other1_entropy_new,other1_sorted_probs),dim=1)
        other2_entropy_features = torch.cat((other2_entropy_new,other2_sorted_probs),dim=1)

        if with_entropy:
            query_average_mid_layer_new = torch.cat((query_average_mid_layer_new,entropy_features),dim=1)
            query_average_last_layer_new = torch.cat((query_average_last_layer_new,entropy_features),dim=1)
            query_last1_token_mid_layer_new = torch.cat((query_last1_token_mid_layer_new,entropy_features),dim=1)
            query_last1_token_last_layer_new = torch.cat((query_last1_token_last_layer_new,entropy_features),dim=1)
            answerm_mid_layer_new = torch.cat((answerm_mid_layer_new,entropy_features),dim=1)
            answerm_last_layer_new = torch.cat((answerm_last_layer_new,entropy_features),dim=1)
            query_answer_ave_mid_new = torch.cat((query_answer_ave_mid_new,entropy_features),dim=1)
            query_answer_ave_last_new = torch.cat((query_answer_ave_last_new,entropy_features),dim=1)
            query_answer_last_token_mid_new = torch.cat((query_answer_last_token_mid_new,entropy_features),dim=1)
            query_answer_last_token_last_new = torch.cat((query_answer_last_token_last_new,entropy_features),dim=1)

            other1_query_average_mid_layer_new = torch.cat((other1_query_average_mid_layer_new,other1_entropy_features),dim=1)
            other1_query_average_last_layer_new = torch.cat((other1_query_average_last_layer_new,other1_entropy_features),dim=1)
            other1_query_last1_token_mid_layer_new = torch.cat((other1_query_last1_token_mid_layer_new,other1_entropy_features),dim=1)
            other1_query_last1_token_last_layer_new = torch.cat((other1_query_last1_token_last_layer_new,other1_entropy_features),dim=1)
            other1_answerm_mid_layer_new = torch.cat((other1_answerm_mid_layer_new,other1_entropy_features),dim=1)
            other1_answerm_last_layer_new = torch.cat((other1_answerm_last_layer_new,other1_entropy_features),dim=1)
            other1_query_answer_ave_mid_new = torch.cat((other1_query_answer_ave_mid_new,other1_entropy_features),dim=1)
            other1_query_answer_ave_last_new = torch.cat((other1_query_answer_ave_last_new,other1_entropy_features),dim=1)
            other1_query_answer_last_token_mid_new = torch.cat((other1_query_answer_last_token_mid_new,other1_entropy_features),dim=1)
            other1_query_answer_last_token_last_new = torch.cat((other1_query_answer_last_token_last_new,other1_entropy_features),dim=1)

            other2_query_average_mid_layer_new = torch.cat((other2_query_average_mid_layer_new,other2_entropy_features),dim=1)
            other2_query_average_last_layer_new = torch.cat((other2_query_average_last_layer_new,other2_entropy_features),dim=1)
            other2_query_last1_token_mid_layer_new = torch.cat((other2_query_last1_token_mid_layer_new,other2_entropy_features),dim=1)
            other2_query_last1_token_last_layer_new = torch.cat((other2_query_last1_token_last_layer_new,other2_entropy_features),dim=1)
            other2_answerm_mid_layer_new = torch.cat((other2_answerm_mid_layer_new,other2_entropy_features),dim=1)
            other2_answerm_last_layer_new = torch.cat((other2_answerm_last_layer_new,other2_entropy_features),dim=1)
            other2_query_answer_ave_mid_new = torch.cat((other2_query_answer_ave_mid_new,other2_entropy_features),dim=1)
            other2_query_answer_ave_last_new = torch.cat((other2_query_answer_ave_last_new,other2_entropy_features),dim=1)
            other2_query_answer_last_token_mid_new = torch.cat((other2_query_answer_last_token_mid_new,other2_entropy_features),dim=1)
            other2_query_answer_last_token_last_new = torch.cat((other2_query_answer_last_token_last_new,other2_entropy_features),dim=1)




        if task_idx==0:
            query_average_mid_layer = query_average_mid_layer_new
            query_average_last_layer = query_average_last_layer_new
            query_last1_token_mid_layer = query_last1_token_mid_layer_new
            query_last1_token_last_layer = query_last1_token_last_layer_new
            answerm_mid_layer = answerm_mid_layer_new
            answerm_last_layer = answerm_last_layer_new
            query_answer_ave_mid = query_answer_ave_mid_new
            query_answer_ave_last = query_answer_ave_last_new
            query_answer_last_token_mid = query_answer_last_token_mid_new
            query_answer_last_token_last = query_answer_last_token_last_new

            other1_query_average_mid_layer = other1_query_average_mid_layer_new
            other1_query_average_last_layer = other1_query_average_last_layer_new
            other1_query_last1_token_mid_layer = other1_query_last1_token_mid_layer_new
            other1_query_last1_token_last_layer = other1_query_last1_token_last_layer_new
            other1_answerm_mid_layer = other1_answerm_mid_layer_new
            other1_answerm_last_layer = other1_answerm_last_layer_new
            other1_query_answer_ave_mid = other1_query_answer_ave_mid_new
            other1_query_answer_ave_last = other1_query_answer_ave_last_new
            other1_query_answer_last_token_mid = other1_query_answer_last_token_mid_new
            other1_query_answer_last_token_last = other1_query_answer_last_token_last_new

            other2_query_average_mid_layer = other2_query_average_mid_layer_new
            other2_query_average_last_layer = other2_query_average_last_layer_new
            other2_query_last1_token_mid_layer = other2_query_last1_token_mid_layer_new
            other2_query_last1_token_last_layer = other2_query_last1_token_last_layer_new
            other2_answerm_mid_layer = other2_answerm_mid_layer_new
            other2_answerm_last_layer = other2_answerm_last_layer_new
            other2_query_answer_ave_mid = other2_query_answer_ave_mid_new
            other2_query_answer_ave_last = other2_query_answer_ave_last_new
            other2_query_answer_last_token_mid = other2_query_answer_last_token_mid_new
            other2_query_answer_last_token_last = other2_query_answer_last_token_last_new

            Y = Y_new
            entropies = entropy_new

            other1_entropies = other1_entropy_new
            other2_entropies = other2_entropy_new

        else:
            query_average_mid_layer = torch.cat((query_average_mid_layer,query_average_mid_layer_new),dim=0)
            query_average_last_layer = torch.cat((query_average_last_layer,query_average_last_layer_new),dim=0)
            query_last1_token_mid_layer = torch.cat((query_last1_token_mid_layer,query_last1_token_mid_layer_new),dim=0)
            query_last1_token_last_layer = torch.cat((query_last1_token_last_layer,query_last1_token_last_layer_new),dim=0)
            answerm_mid_layer = torch.cat((answerm_mid_layer,answerm_mid_layer_new),dim=0)
            answerm_last_layer = torch.cat((answerm_last_layer,answerm_last_layer_new),dim=0)
            query_answer_ave_mid = torch.cat((query_answer_ave_mid,query_answer_ave_mid_new),dim=0)
            query_answer_ave_last = torch.cat((query_answer_ave_last,query_answer_ave_last_new),dim=0)
            query_answer_last_token_mid = torch.cat((query_answer_last_token_mid,query_answer_last_token_mid_new),dim=0)
            query_answer_last_token_last = torch.cat((query_answer_last_token_last,query_answer_last_token_last_new),dim=0)

            other1_query_average_mid_layer = torch.cat((other1_query_average_mid_layer,other1_query_average_mid_layer_new),dim=0)
            other1_query_average_last_layer = torch.cat((other1_query_average_last_layer,other1_query_average_last_layer_new),dim=0)
            other1_query_last1_token_mid_layer = torch.cat((other1_query_last1_token_mid_layer,other1_query_last1_token_mid_layer_new),dim=0)
            other1_query_last1_token_last_layer = torch.cat((other1_query_last1_token_last_layer,other1_query_last1_token_last_layer_new),dim=0)
            other1_answerm_mid_layer = torch.cat((other1_answerm_mid_layer,other1_answerm_mid_layer_new),dim=0)
            other1_answerm_last_layer = torch.cat((other1_answerm_last_layer,other1_answerm_last_layer_new),dim=0)
            other1_query_answer_ave_mid = torch.cat((other1_query_answer_ave_mid,other1_query_answer_ave_mid_new),dim=0)
            other1_query_answer_ave_last = torch.cat((other1_query_answer_ave_last,other1_query_answer_ave_last_new),dim=0)
            other1_query_answer_last_token_mid = torch.cat((other1_query_answer_last_token_mid,other1_query_answer_last_token_mid_new),dim=0)
            other1_query_answer_last_token_last = torch.cat((other1_query_answer_last_token_last,other1_query_answer_last_token_last_new),dim=0)

            other2_query_average_mid_layer = torch.cat((other2_query_average_mid_layer,other2_query_average_mid_layer_new),dim=0)
            other2_query_average_last_layer = torch.cat((other2_query_average_last_layer,other2_query_average_last_layer_new),dim=0)
            other2_query_last1_token_mid_layer = torch.cat((other2_query_last1_token_mid_layer,other2_query_last1_token_mid_layer_new),dim=0)
            other2_query_last1_token_last_layer = torch.cat((other2_query_last1_token_last_layer,other2_query_last1_token_last_layer_new),dim=0)
            other2_answerm_mid_layer = torch.cat((other2_answerm_mid_layer,other2_answerm_mid_layer_new),dim=0)
            other2_answerm_last_layer = torch.cat((other2_answerm_last_layer,other2_answerm_last_layer_new),dim=0)
            other2_query_answer_ave_mid = torch.cat((other2_query_answer_ave_mid,other2_query_answer_ave_mid_new),dim=0)
            other2_query_answer_ave_last = torch.cat((other2_query_answer_ave_last,other2_query_answer_ave_last_new),dim=0)
            other2_query_answer_last_token_mid = torch.cat((other2_query_answer_last_token_mid,other2_query_answer_last_token_mid_new),dim=0)
            other2_query_answer_last_token_last = torch.cat((other2_query_answer_last_token_last,other2_query_answer_last_token_last_new),dim=0)


            Y = torch.cat((Y,Y_new),dim=0)
            entropies = torch.cat((entropies,entropy_new),dim=0)

            other1_entropies = torch.cat((other1_entropies,other1_entropy_new),dim=0)
            other2_entropies = torch.cat((other2_entropies,other2_entropy_new),dim=0)

        
    origin_name_list = ['query-ave-mid-layer','query-ave-last-layer','query-last-token-mid-layer','query-last-token-last-layer','answerm-mid-layer','answerm-last-layer']
    
    other1_name_list = [other1_name+name for name in origin_name_list]
    other2_name_list = [other2_name+name for name in origin_name_list]
    name_list = origin_name_list+other1_name_list+other2_name_list
    
    data_list = []
    data_list.append(query_average_mid_layer)
    data_list.append(query_average_last_layer)
    data_list.append(query_last1_token_mid_layer)
    data_list.append(query_last1_token_last_layer)
    data_list.append(answerm_mid_layer)
    data_list.append(answerm_last_layer)

    data_list.append(other1_query_average_mid_layer)
    data_list.append(other1_query_average_last_layer)
    data_list.append(other1_query_last1_token_mid_layer)
    data_list.append(other1_query_last1_token_last_layer)
    data_list.append(other1_answerm_mid_layer)
    data_list.append(other1_answerm_last_layer)

    data_list.append(other2_query_average_mid_layer)
    data_list.append(other2_query_average_last_layer)
    data_list.append(other2_query_last1_token_mid_layer)
    data_list.append(other2_query_last1_token_last_layer)
    data_list.append(other2_answerm_mid_layer)
    data_list.append(other2_answerm_last_layer)


    name_list.append("entropy")
    data_list.append(query_average_mid_layer[:,-5].reshape(-1,1))

    name_list.append(other1_name+"entropy")
    data_list.append(other1_query_average_mid_layer[:,-5].reshape(-1,1))

    name_list.append(other2_name+"entropy")
    data_list.append(other2_query_average_mid_layer[:,-5].reshape(-1,1))

    name_list.append("max prob")
    data_list.append(query_average_mid_layer[:,-4].reshape(-1,1))

    name_list.append(other1_name+"max prob")
    data_list.append(other1_query_average_mid_layer[:,-4].reshape(-1,1))

    name_list.append(other2_name+"max prob")
    data_list.append(other2_query_average_mid_layer[:,-4].reshape(-1,1))

    name_list.append("entropy-supervised")
    data_list.append(query_average_mid_layer[:,-5:])

    name_list.append(other1_name+"entropy-supervised")
    data_list.append(other1_query_average_mid_layer[:,-5:])

    name_list.append(other2_name+"entropy-supervised")
    data_list.append(other2_query_average_mid_layer[:,-5:])

    name_list.append("query-ans-ave-mid")
    data_list.append(query_answer_ave_mid)

    name_list.append(other1_name+"query-ans-ave-mid")
    data_list.append(other1_query_answer_ave_mid)

    name_list.append(other2_name+"query-ans-ave-mid")
    data_list.append(other2_query_answer_ave_mid)

    name_list.append("query-ans-ave-last")
    data_list.append(query_answer_ave_last)

    name_list.append(other1_name+"query-ans-ave-last")
    data_list.append(other1_query_answer_ave_last)

    name_list.append(other2_name+"query-ans-ave-last")
    data_list.append(other2_query_answer_ave_last)

    name_list.append("query-ans-last-token-mid")
    data_list.append(query_answer_last_token_mid)

    name_list.append(other1_name+"query-ans-last-token-mid")
    data_list.append(other1_query_answer_last_token_mid)

    name_list.append(other2_name+"query-ans-last-token-mid")
    data_list.append(other2_query_answer_last_token_mid)

    name_list.append("query-ans-last-token-last")
    data_list.append(query_answer_last_token_last)

    name_list.append(other1_name+"query-ans-last-token-last")
    data_list.append(other1_query_answer_last_token_last)

    name_list.append(other2_name+"query-ans-last-token-last")
    data_list.append(other2_query_answer_last_token_last)


    if phase=="validation":
        ask4conf_score = load_ask4conf_score(dataset_name="MMLU",model_type=model_name,available_idxs=list(range(query_answer_last_token_last.shape[0])),MMLU_TASKS=MMLU_TASKS)
    else:
        ask4conf_score = None

    return data_list,name_list,Y,ask4conf_score



def get_index_of_valid_X(dataset_name,model_type,phase="test"):

    SU_KEY = 'semantic_entropy'
    dataset_name = dataset_name

    if dataset_name.startswith("wmt"):
        metric = 'bleu'
    else:
        metric = 'rouge1_most'

    output_dir = "./test_output/"+dataset_name+"/"+model_type+"/"
    data_json_path = output_dir+dataset_name+"_extend.json"
    if dataset_name.startswith("wmt"):
        mrouge_path = output_dir+dataset_name+"_mextend_bleu.json"
    else:
        mrouge_path = output_dir+dataset_name+"_mextend_rouge.json"
    SU_path = output_dir+dataset_name+"_semantic_entropy.json"

    with open(data_json_path) as fr:
        data = json.load(fr)

    with open(mrouge_path) as fr:
        mrouge = json.load(fr)

    if phase=="test":
        with open(SU_path) as fr:
            SU_data = json.load(fr)
    else:
        SU_data = data

    available_idxs = []
    mrouges = []
    SU_scores = []
    print("here")
    for d_idx,d in enumerate(data):
   
        if metric in mrouge[d_idx]:
            if phase!="test" or (SU_KEY in SU_data[d_idx]):# make sure SU has been calculated, and the metric has been calculated
                
                
                #if d[QUERY_STR] == mrouge[d_idx][QUERY_STR]: # check they are the same question

                available_idxs.append(d_idx)
                mrouges.append(mrouge[d_idx][metric])
                if SU_KEY in SU_data[d_idx]:
                    SU_scores.append(-SU_data[d_idx][SU_KEY])

            
    print("load "+dataset_name+" "+str(len(mrouges)))

    if dataset_name.startswith("triviaqa") or dataset_name.startswith("coqa"):
        if phase=="test":
            available_idxs = available_idxs[:2000]
            mrouges = mrouges[:2000]
            SU_scores = SU_scores[:2000]
        else:
            available_idxs = available_idxs[2000:]
            mrouges = mrouges[2000:]
    if phase=="test":
        try:
            ask4conf_score = load_ask4conf_score(dataset_name=dataset_name,model_type=model_type,available_idxs=available_idxs)
        except:
            ask4conf_score = None
    else:
        ask4conf_score = None

    return available_idxs,mrouges,SU_scores,ask4conf_score


def load_X_Y_with_phase(dataset_name,model_type,phase="test",with_entropy=True):
    output_dir = "./test_output/"+dataset_name+"/"+model_type+"/"
    

    print(output_dir)
    available_idxs,mrouges,SU_scores,ask4conf_score = get_index_of_valid_X(dataset_name,model_type,phase=phase)
    # prepare X
    if model_type=="llama_2_7b":
        layer_list = [16,32]
        layer_list_other1 = [14,28] # gemma 7b
        layer_list_other2 = [9,18] # gemma 2b
        other1_name = 'other-7B-'
        other2_name = 'other-2B-'
        other1_output_dir = "./test_output/"+dataset_name+"/gemma_7b/"
        other2_output_dir = "./test_output/"+dataset_name+"/gemma_2b/"


    elif model_type=="gemma_7b":
        layer_list = [14,28]
        layer_list_other1 = [16,32] # llama 2 7b
        layer_list_other2 = [20,40] # llama 2 13b
        other1_name = 'other-7B-'
        other2_name = 'other-13B-'
        other1_output_dir = "./test_output/"+dataset_name+"/llama_2_7b/"
        other2_output_dir = "./test_output/"+dataset_name+"/llama_2_13b/"


    def get_file_name_list(layer_list,is_cross=False):
        query_average_mid_layer_name = 'query_average_layer_'+str(layer_list[0])+'.pt'
        query_average_last_layer_name = 'query_average_layer_'+str(layer_list[1])+'.pt'
        query_last1_token_mid_layer_name = 'query_last_1_token_layer_'+str(layer_list[0])+'.pt'
        query_last1_token_last_layer_name = 'query_last_1_token_layer_'+str(layer_list[1])+'.pt'
        if not is_cross:
            answerm_mid_layer_name = 'answerm_average_layer_'+str(layer_list[0])+'.pt'  
            answerm_last_layer_name = 'answerm_average_layer_'+str(layer_list[1])+'.pt'
            answer_last1_token_mid_layer_name = 'answerm_last_1_token_layer_'+str(layer_list[0])+'.pt'
            answer_last1_token_last_layer_name = 'answerm_last_1_token_layer_'+str(layer_list[1])+'.pt'
            file_name_list = [query_average_mid_layer_name,query_average_last_layer_name,query_last1_token_mid_layer_name,query_last1_token_last_layer_name,answerm_mid_layer_name,answerm_last_layer_name,answer_last1_token_mid_layer_name,answer_last1_token_last_layer_name]
        else:
            other_answerm_mid_layer_name = 'cross_average_layer_'+str(layer_list[0])+'.pt'
            other_answerm_last_layer_name = 'cross_average_layer_'+str(layer_list[1])+'.pt'
            other_answer_last1_token_mid_layer_name = 'cross_last_1_token_layer_'+str(layer_list[0])+'.pt'
            other_answer_last1_token_last_layer_name = 'cross_last_1_token_layer_'+str(layer_list[1])+'.pt'
            file_name_list = [query_average_mid_layer_name,query_average_last_layer_name,query_last1_token_mid_layer_name,query_last1_token_last_layer_name,other_answerm_mid_layer_name,other_answerm_last_layer_name,other_answer_last1_token_mid_layer_name,other_answer_last1_token_last_layer_name]

        return file_name_list

    file_name_list = get_file_name_list(layer_list)
    other1_file_name_list = get_file_name_list(layer_list_other1,is_cross=True)
    other2_file_name_list = get_file_name_list(layer_list_other2,is_cross=True)

    def load_file(file_name_list,output_dir,available_idxs):
        data_list = []
        for file_name in file_name_list:
            data = torch.load(output_dir+file_name)
            data = data[available_idxs].squeeze()
            data_list.append(data)
        return data_list
    
    X_list = load_file(file_name_list,output_dir,available_idxs)
    other1_X_list = load_file(other1_file_name_list,other1_output_dir,available_idxs)
    other2_X_list = load_file(other2_file_name_list,other2_output_dir,available_idxs)

    X_list.append(torch.cat((X_list[0],X_list[4]),dim=1)) # query_ans_ave_mid_layer
    X_list.append(torch.cat((X_list[1],X_list[5]),dim=1)) # query_ans_ave_last_layer
    X_list.append(torch.cat((X_list[2],X_list[6]),dim=1)) # query_ans_last_token_mid_layer
    X_list.append(torch.cat((X_list[3],X_list[7]),dim=1)) # query_ans_last_token_last_layer
    
    other1_X_list.append(torch.cat((other1_X_list[0],other1_X_list[4]),dim=1)) # query_ans_ave_mid_layer
    other1_X_list.append(torch.cat((other1_X_list[1],other1_X_list[5]),dim=1)) # query_ans_ave_last_layer
    other1_X_list.append(torch.cat((other1_X_list[2],other1_X_list[6]),dim=1)) # query_ans_last_token_mid_layer
    other1_X_list.append(torch.cat((other1_X_list[3],other1_X_list[7]),dim=1)) # query_ans_last_token_last_layer

    other2_X_list.append(torch.cat((other2_X_list[0],other2_X_list[4]),dim=1)) # query_ans_ave_mid_layer
    other2_X_list.append(torch.cat((other2_X_list[1],other2_X_list[5]),dim=1)) # query_ans_ave_last_layer
    other2_X_list.append(torch.cat((other2_X_list[2],other2_X_list[6]),dim=1)) # query_ans_last_token_mid_layer
    other2_X_list.append(torch.cat((other2_X_list[3],other2_X_list[7]),dim=1)) # query_ans_last_token_last_layer

    


    name_list = ['query-ave mid layer', 'query-ave last layer', 'query last-token mid layer', 'query last-token last layer',
                    'answerm-ave mid layer', 'answerm-ave last layer', 'answerm last-token mid layer', 'answerm last-token last layer','query-ans-ave-mid','query-ans-ave-last','query-ans-last-token-mid','query-ans-last-token-last']
    other1_name_list = [other1_name+name for name in name_list]
    other2_name_list = [other2_name+name for name in name_list]
    name_list = name_list + other1_name_list + other2_name_list

    data_list = X_list + other1_X_list + other2_X_list

    
    
    # load entropy data
    query_entropy_name = 'query_entropies.pt'
    answerm_entropy_name = 'answerm_entropies.pt'
    query_probs_name = 'query_probs.pt'
    answerm_probs_name = 'answerm_probs.pt'

    other_answerm_entropy_name = 'cross_entropies.pt'
    other_answerm_probs_name = 'cross_probs.pt'

    query_entropies = torch.load(output_dir+query_entropy_name)
    answerm_entropies = torch.load(output_dir+answerm_entropy_name)
    query_probs = torch.load(output_dir+query_probs_name)
    answerm_probs = torch.load(output_dir+answerm_probs_name)

    other1_query_entropies = torch.load(other1_output_dir+query_entropy_name)
    other1_answerm_entropies = torch.load(other1_output_dir+other_answerm_entropy_name)
    other1_query_probs = torch.load(other1_output_dir+query_probs_name)
    other1_answerm_probs = torch.load(other1_output_dir+other_answerm_probs_name)

    other2_query_entropies = torch.load(other2_output_dir+query_entropy_name)
    other2_answerm_entropies = torch.load(other2_output_dir+other_answerm_entropy_name)
    other2_query_probs = torch.load(other2_output_dir+query_probs_name)
    other2_answerm_probs = torch.load(other2_output_dir+other_answerm_probs_name)


    query_entropies = query_entropies[available_idxs]
    answerm_entropies = answerm_entropies[available_idxs]
    query_probs = query_probs[available_idxs]
    answerm_probs = answerm_probs[available_idxs]

    other1_query_entropies = other1_query_entropies[available_idxs]
    other1_answerm_entropies = other1_answerm_entropies[available_idxs]
    other1_query_probs = other1_query_probs[available_idxs]
    other1_answerm_probs = other1_answerm_probs[available_idxs]

    other2_query_entropies = other2_query_entropies[available_idxs]
    other2_answerm_entropies = other2_answerm_entropies[available_idxs]
    other2_query_probs = other2_query_probs[available_idxs]
    other2_answerm_probs = other2_answerm_probs[available_idxs]
    

    query_entropies = torch.cat((query_entropies,query_probs),dim=1)
    answerm_entropies = torch.cat((answerm_entropies,answerm_probs),dim=1)

    other1_query_entropies = torch.cat((other1_query_entropies,other1_query_probs),dim=1)
    other1_answerm_entropies = torch.cat((other1_answerm_entropies,other1_answerm_probs),dim=1)

    other2_query_entropies = torch.cat((other2_query_entropies,other2_query_probs),dim=1)
    other2_answerm_entropies = torch.cat((other2_answerm_entropies,other2_answerm_probs),dim=1)



    if with_entropy:
        # concatenate extropy_data to all the layer data
        for idx in range(len(data_list)):
            if name_list[idx].startswith('query-ans'):
                data_list[idx] = torch.cat((data_list[idx],query_entropies,answerm_entropies),dim=1)

            elif name_list[idx].startswith('query'):
                data_list[idx]= torch.cat((data_list[idx],query_entropies),dim=1)

            elif name_list[idx].startswith('answerm'):
                data_list[idx]= torch.cat((data_list[idx],answerm_entropies),dim=1)

            elif name_list[idx].startswith(other1_name+'query-ans'):
                data_list[idx] = torch.cat((data_list[idx],other1_query_entropies,other1_answerm_entropies),dim=1)

            elif name_list[idx].startswith(other1_name+'query'):
                data_list[idx]= torch.cat((data_list[idx],other1_query_entropies),dim=1)

            elif name_list[idx].startswith(other1_name+'answerm'):
                data_list[idx]= torch.cat((data_list[idx],other1_answerm_entropies),dim=1)

            elif name_list[idx].startswith(other2_name+'query-ans'):
                data_list[idx] = torch.cat((data_list[idx],other2_query_entropies,other2_answerm_entropies),dim=1)
            elif name_list[idx].startswith(other2_name+'query'):
                data_list[idx]= torch.cat((data_list[idx],other2_query_entropies),dim=1)
            elif name_list[idx].startswith(other2_name+'answerm'):
                data_list[idx]= torch.cat((data_list[idx],other2_answerm_entropies),dim=1)



    y = mrouges

    for column_idx in range(answerm_entropies.shape[1]):
        data_list.append(answerm_entropies[:,column_idx].reshape(-1,1))
    data_list.append(query_entropies)
    data_list.append(answerm_entropies)
    data_list.append(torch.cat((query_entropies,answerm_entropies),dim=1))
        
    extend_name_list = ["Max Entropy","Min Entropy", "Entropy Avg","Entropy Std",'Max Prob','Min Prob','Prob Mean','Prob Std','Log Prob Mean','Log Prob Std','supervised-query-entropy','supervised-answer-entropy','supervised-query-answer-entropy']
    name_list.extend(extend_name_list)

    for column_idx in range(other1_answerm_entropies.shape[1]):
        data_list.append(other1_answerm_entropies[:,column_idx].reshape(-1,1))
    data_list.append(other1_query_entropies)
    data_list.append(other1_answerm_entropies)
    data_list.append(torch.cat((other1_query_entropies,other1_answerm_entropies),dim=1))

    name_list.extend([other1_name+name for name in extend_name_list])

    for column_idx in range(other2_answerm_entropies.shape[1]):
        data_list.append(other2_answerm_entropies[:,column_idx].reshape(-1,1))
    data_list.append(other2_query_entropies)
    data_list.append(other2_answerm_entropies)
    data_list.append(torch.cat((other2_query_entropies,other2_answerm_entropies),dim=1))

    name_list.extend([other2_name+name for name in extend_name_list])

    
    
    return data_list, name_list, y, SU_scores, ask4conf_score

def load_X_Y(dataset_name,model_type,with_entropy=True):

    # load test data
    if dataset_name == "coqa":
        test_dataset_name = "coqa__train"
        train_dataset_name = "coqa__train"
    elif dataset_name == "triviaqa":
        test_dataset_name = "triviaqa__train"
        train_dataset_name = "triviaqa__train"
    elif dataset_name == "wmt":
        test_dataset_name = "wmt__test"
        train_dataset_name = "wmt__train"
    else:
        # raise error to hint the name should be in list['coqa','triviaqa','wmt']
        raise ValueError("dataset name not supported, should be in ['coqa','triviaqa','wmt']")
    
    print("go to load test data")
    # load test data
    data_test_list,name_test_list,y_test,SU_test,ask4conf_test = load_X_Y_with_phase(test_dataset_name,model_type,phase="test",with_entropy=with_entropy)
    print("go to load training data")
    # load train data
    data_train_list,name_train_list,y_train,_ ,_= load_X_Y_with_phase(train_dataset_name,model_type,phase="train",with_entropy=with_entropy)

    return data_train_list,data_test_list,name_train_list,y_train,y_test,SU_test,ask4conf_test



def load_ask4conf_score(dataset_name: Literal["coqa__test", "triviaqa__train", "wmt__test", "MMLU", "coqa__train"],
                        model_type: Literal["llama_2_7b", "gemma_7b"],
                        available_idxs: list[int],
                        MMLU_TASKS: Literal['all', 'Group1', 'Group2',]='all',
                        ) -> pd.Series:
    MMLU_TASK_list = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']
    
    
    if MMLU_TASKS=='Group1':
        MMLU_TASKS = MMLU_TASK_list[:40]
    elif MMLU_TASKS=='Group2':
        MMLU_TASKS = MMLU_TASK_list[40:]
    elif MMLU_TASKS=='all':
        MMLU_TASKS = MMLU_TASK_list
    else:
        raise ValueError("MMLU_TASKS should be in ['Group1','Group2','all']")
    
    root = "test_output/ask4conf/"
    files = {
        "coqa__test": ["coqa__test.jsonl"],
        "coqa__train": ["coqa__train.jsonl"],
        "triviaqa__train": ["triviaqa__triviaqa__train.jsonl"],
        "wmt__test": ["wmt__wmt__test.jsonl"],
        "MMLU": [f"mmlu__mmlu__{task}__validation.jsonl" for task in MMLU_TASKS]
    }

    files_to_read = [Path(root) / model_type / fname for fname in files[dataset_name]]
    
    dfs = []
    for f in files_to_read:
        try:
            df = pd.read_json(str(f), lines=True)
            dfs.append(df)
        except Exception as e:
            print(f"{f}")
        
    df = pd.concat(dfs, axis=0).reset_index(drop=True)
    return df["prob"].iloc[[_ for _ in available_idxs if _ < df["prob"].index.stop]]