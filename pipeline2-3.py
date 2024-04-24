# %%
import json
import torch

from funs_load_X_and_Y import load_X_Y
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="triviaqa")
parser.add_argument("--model_type", type=str, default="gemma_7b")
args = parser.parse_args()

dataset_name = args.dataset_name #["triviaqa","coqa","wmt"]
model_type = args.model_type
with_entropy = True
maintain_all_entropies = True

if model_type=="llama_2_7b":
    num_dim = 4096
    other1_num_dim = 3072 # gemma 7b
    other2_num_dim = 2048 # gemma 2b
    other1_name = 'other-7B-'
    other2_name = 'other-2B-'
elif model_type=="gemma_7b":
    num_dim = 3072
    other1_num_dim = 4096 # llama2 7b
    other2_num_dim = 5120 # llama2 13b
    other1_name = 'other-7B-'
    other2_name = 'other-13B-'

data_train_list,data_test_list,name_list,y_train,y_test,SU_test,ask4conf_test = load_X_Y(dataset_name,model_type,with_entropy=with_entropy)
# data_list, name_list, y, SU_scores = load_X_Y(dataset_name,model_type,with_entropy=with_entropy)
output_dir = "./test_output/"+dataset_name+"/"+model_type+"/"
if with_entropy:
    output_dir += "P2_with_entropy/"
    if maintain_all_entropies:
        output_dir += "maintain_all_entropies/"
else:
    output_dir += "P2_without_entropy/"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

threshold = 0.3



# %%
"""
alg_idx_list = [8,9,10,11,12,14,16,20,24]
data_train_list = [data_train_list[i] for i in alg_idx_list]
data_test_list = [data_test_list[i] for i in alg_idx_list]
name_list = [name_list[i] for i in alg_idx_list]
"""

# %%
# prepare Y
import pandas as pd
Y_train = pd.DataFrame(y_train)
Y_train = Y_train.reset_index(drop=True)
Y_test = pd.DataFrame(y_test)
Y_test = Y_test.reset_index(drop=True)


# %%
# do lasso regression and select top 100 features
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, Lasso
from funs_feature_selection import lasso_select_k_features
from joblib import Parallel, delayed
import numpy as np

maintain_all_entropies = True
features_from_saved_list = []
feature_already_selected_list = []
if maintain_all_entropies:
    for data_idx in range(len(data_train_list)):
        name = name_list[data_idx]
        # check if there are already features selected
        feature_file_name = output_dir + name + "_selected_features.json"
        if os.path.exists(feature_file_name):
            with open(feature_file_name, 'r') as f:
                feature_already_selected = json.load(f)
            feature_already_selected_list.append(feature_already_selected)
            features_from_saved_list.append(True)
            continue

        features_from_saved_list.append(False)
        if data_train_list[data_idx].shape[1]>min(num_dim,other1_num_dim,other2_num_dim):
            if name_list[data_idx].startswith(other1_name):
                if name_list[data_idx].startswith(other1_name+'query-ans'):
                    feature_already_selected = list(range(2*other1_num_dim,data_train_list[data_idx].shape[1]))
                else:
                    feature_already_selected = list(range(other1_num_dim,data_train_list[data_idx].shape[1]))
            elif name_list[data_idx].startswith(other2_name):
                if name_list[data_idx].startswith(other2_name+'query-ans'):
                    feature_already_selected = list(range(2*other2_num_dim,data_train_list[data_idx].shape[1]))
                else:
                    feature_already_selected = list(range(other2_num_dim,data_train_list[data_idx].shape[1]))
            else:
                if name_list[data_idx].startswith('query-ans'):
                    feature_already_selected = list(range(2*num_dim,data_train_list[data_idx].shape[1]))
                else:
                    feature_already_selected = list(range(num_dim,data_train_list[data_idx].shape[1]))
        else:
            feature_already_selected = list(range(data_train_list[data_idx].shape[1]))
            
        feature_already_selected_list.append(feature_already_selected)
    
lasso_feature_idx_list = Parallel(n_jobs=-1, verbose=1)(delayed(lasso_select_k_features)(data_train_list[idx], Y_train, k=100,features_already_selected=feature_already_selected_list[idx],features_from_saved=features_from_saved_list[idx]) for idx in range(len(data_train_list)))

    
for idx in range(len(data_train_list)):
    feature_already_selected_list[idx] += list(lasso_feature_idx_list[idx])

# %%
# do mutual information feature selection

from funs_feature_selection import mutual_info_select_k_features

from joblib import Parallel, delayed


mutual_features_list = Parallel(n_jobs=-1, verbose=1)(delayed(mutual_info_select_k_features)(data_train_list[idx], Y_train, k=100,features_already_selected=feature_already_selected_list[idx],features_from_saved=features_from_saved_list[idx]) for idx in range(len(data_train_list)))

for idx in range(len(data_train_list)):
    feature_already_selected_list[idx] += list(mutual_features_list[idx])


# %%

from funs_feature_selection import correlation_select_k_features
correlation_features_list = Parallel(n_jobs=-1, verbose=1)(delayed(correlation_select_k_features)(data_train_list[idx], Y_train, k=100,features_already_selected=feature_already_selected_list[idx],features_from_saved=features_from_saved_list[idx]) for idx in range(len(data_train_list)))

for idx in range(len(data_train_list)):
    feature_already_selected_list[idx] += list(correlation_features_list[idx])

    

# %%
# save selected features

import json
for idx in range(len(feature_already_selected_list)):
    name = name_list[idx]
    save_path = output_dir + name + "_selected_features.json"
    if not features_from_saved_list[idx]:
        with open(save_path, 'w') as f:
            json.dump([int(feature) for feature in feature_already_selected_list[idx]], f)



# %%
# random tree regression

from sklearn.ensemble import RandomForestRegressor
import pickle



def random_forest(X_train,Y_train,feature_already_selected,data_name):
    rf_file_name = output_dir + data_name + "_random_forest.pkl"
    if os.path.exists(rf_file_name):
        with open(rf_file_name, 'rb') as f:
            reg = pickle.load(f)
        return reg
    if len(feature_already_selected)<100:
        reg = RandomForestRegressor(n_estimators=100, random_state=0, max_depth=4, verbose=2)
    else:
        reg = RandomForestRegressor(n_estimators=150, random_state=0, max_depth=8, verbose=2, max_features=45)
    
    reg.fit(X_train[:,feature_already_selected], Y_train)

    with open(rf_file_name, 'wb') as f:
        pickle.dump(reg, f)
    return reg

reg_list = Parallel(n_jobs=-1, verbose=1)(delayed(random_forest)(data_train_list[idx].detach().cpu().numpy(), Y_train,feature_already_selected_list[idx],name_list[idx]) for idx in range(len(data_train_list)))
    


# %%
# get the output of test data
# import crossentropy loss for the output
import torch.nn as nn
import torch
import numpy as np
import pandas as pd

# get the output of test data
output_list = []
for idx,X_test in enumerate(data_test_list):
    if X_test.shape[1]>1:
        test_data = X_test[:,feature_already_selected_list[idx]]
        # check if there is missing value or nan in test_data
        if np.isnan(test_data).any():
            print(f"nan in {name_list[idx]}")
            

        output = reg_list[idx].predict(X_test[:,feature_already_selected_list[idx]])
    else:
        output = -X_test.reshape(-1)
    output_list.append(output)
    


# %%

# calculate the AUROC
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

Y_test_discrete = [1 if y > threshold else 0 for y in y_test]
aucs = []
fprs = []
tprs = []
for idx,output in enumerate(output_list):
    fpr, tpr, _ = roc_curve(Y_test_discrete, output)
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    fprs.append(fpr)
    tprs.append(tpr)
    
SU_fpr, SU_tpr, SU_thresholds = roc_curve(Y_test_discrete, SU_test)
SU_roc_auc = auc(SU_fpr, SU_tpr)
fprs.append(SU_fpr)
tprs.append(SU_tpr)
aucs.append(SU_roc_auc)
name_list.append('Semantic entropy')

if ask4conf_test is not None:
    ask4conf_fpr, ask4conf_tpr, ask4conf_thresholds = roc_curve(Y_test_discrete, ask4conf_test)
    ask4conf_roc_auc = auc(ask4conf_fpr, ask4conf_tpr)
    fprs.append(ask4conf_fpr)
    tprs.append(ask4conf_tpr)
    aucs.append(ask4conf_roc_auc)
    name_list.append('Ask4Conf')

plt.figure()

for idx in range(len(aucs)):
    plt.plot(fprs[idx], tprs[idx], lw=1, label=f'{name_list[idx]}, area = {aucs[idx]:.3f}')


# plt.plot(SU_fpr, SU_tpr, color='grey',linestyle='--', lw=1, label='Semantic entropy (area = %0.3f)' % SU_roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
dataset_formal_name = " "
if dataset_name == "triviaqa__train":
    dataset_formal_name = f'TriviaQA (#train: {int(len(y_train))} ; #test: {int(len(y_test))})'
elif dataset_name == "coqa__train":
    dataset_formal_name = f'CoQA (#train: {int(len(y_train))} ; #test: {int(len(y_test))})'
if with_entropy:
    title_str = 'ROC on test data of '+dataset_formal_name+' with entropies'
else:
    title_str = 'ROC on test data of '+dataset_formal_name+ ' without entropies'
plt.title(title_str)
plt.legend(loc="lower right")
plt.savefig(output_dir+"p2"+title_str+".pdf")

# %%
# save the results
file_name = output_dir + "p2_results.csv"
results = pd.DataFrame({"model":name_list,"auc":aucs})
results.to_csv(file_name,index=False)



