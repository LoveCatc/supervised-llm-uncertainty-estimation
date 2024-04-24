# %%
from funs_load_X_and_Y import load_MMLU_X_Y
import torch

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model_type', type=str, default="gemma_7b", help='model type')
args = parser.parse_args()


with_entropy = True
maintain_all_entropies = True
pipeline_type = "P2"

model_type = args.model_type

with_entropy=True

data_test_list,name_list,y_test,ask4conf_test = load_MMLU_X_Y(phase='validation',model_name=model_type,with_entropy=with_entropy,MMLU_TASKS = 'all')


output_dir = "./test_output/MMLU/"+model_type+"/all/"
if with_entropy:
    output_dir += "P2_with_entropy/"
else:
    output_dir += "P2_without_entropy/"

y_test = [1 if y>0.3 else 0 for y in y_test]
y_test = torch.tensor(y_test)

ask4conf_test = torch.tensor(ask4conf_test)


data_test_list.append(ask4conf_test)
name_list.append("Ask4conf")



# %%
# load all the selected features
feature_name_list = []
random_forest_list = []
selected_features = []
# get all the files end with "_selected_features.json" under ood_output_dir
import os
import json
import pickle
for file in os.listdir(output_dir):
    if file.endswith("_selected_features.json"):
        
        rf_name = file[:-len("_selected_features.json")]+"_random_forest.pkl"
        # check if the rf exists
        if not os.path.exists(output_dir+rf_name):
            continue

        feature_name_list.append(file[:-len("_selected_features.json")])
        with open(output_dir+file) as f:
            selected_features.append(json.load(f))
        with open(output_dir+rf_name,"rb") as f:
            random_forest_list.append(pickle.load(f))



# %%
# remain only id_test_list when name_list is in feature_name_list
selected_id_test_list = []
selected_name_list = []
for name in feature_name_list:
    selected_name_list.append(name)
    name_idx = name_list.index(name)
    selected_id_test_list.append(data_test_list[name_idx])


# %%
for idx in range(len(selected_name_list)):
    print(selected_name_list[idx])
    print(selected_features[idx])
    print(selected_id_test_list[idx].shape)


# %%
y_pred_list = []
for idx in range(len(random_forest_list)):
    # feature selection
    selected_id_test_list[idx] = selected_id_test_list[idx][:,selected_features[idx]]
    if len(selected_id_test_list[idx].shape)==1:
        selected_id_test_list[idx] = selected_id_test_list[idx].reshape(-1,1)
    
    # forward random forest
    if selected_id_test_list[idx].shape[1]>1:
        y_pred = random_forest_list[idx].predict(selected_id_test_list[idx])
    else:
        y_pred = -selected_id_test_list[idx].reshape(-1)
    y_pred_list.append(y_pred)


selected_name_list.append("ask4conf")


y_pred_list.append(ask4conf_test)

#%% calculate the roc for y_pred, if it is <0.5, flip the y_pred
from sklearn.metrics import roc_auc_score
for idx in range(len(y_pred_list)):
    roc = roc_auc_score(y_test,y_pred_list[idx])
    if roc<0.5:
        y_pred_list[idx] = -y_pred_list[idx]

# %% normalize the y_pred to [0,1]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
y_pred_list_normalized = []
for y_pred in y_pred_list:
    y_pred_normalized = scaler.fit_transform(y_pred.reshape(-1,1)).reshape(-1)
    y_pred_list_normalized.append(y_pred_normalized)
    print(min(y_pred_normalized),max(y_pred_normalized))

# convert y_pred to logits
import numpy as np
y_pred_logits_list = []
for idx in range(len(y_pred_list_normalized)):
    y_pred_logits = y_pred_list_normalized[idx]
    y_pred_logits = (y_pred_logits+1e-15)/(1-y_pred_logits+1e-15)
    y_pred_logits = np.log(y_pred_logits)
    y_pred_logits = torch.tensor(y_pred_logits)
    y_pred_logits_list.append(y_pred_logits)


# %% split the data: 30% for calibration, 70% for test
from sklearn.model_selection import train_test_split
#random_state = 0
#cal_idxs, test_idxs = train_test_split(range(len(y_test)), test_size=0.7, random_state=random_state)
np.random.seed(0)
idxs = np.arange(len(y_test))
np.random.shuffle(idxs)
calibration_ratio = 0.3
cal_idxs = idxs[:int(len(idxs)*calibration_ratio)]
test_idxs = idxs[int(len(idxs)*calibration_ratio):]
y_pred_cal_list = []
y_pred_test_list = []
y_cal = y_test[cal_idxs]
y_test = y_test[test_idxs]
for y_pred in y_pred_logits_list:
    y_pred_cal_list.append(y_pred[cal_idxs])
    y_pred_test_list.append(y_pred[test_idxs])

from calibration_metrics import calculate_ece,calculate_nll,calculate_nll_ece,calculate_Brier
from temperature_scaling_self import ModelWithTemperature
from plat_scaling_self import PlatScaling

def evaluate_calibration(y_logits,labels):
    nll,ece = calculate_nll_ece(y_logits,labels)
    brier = calculate_Brier(y_logits,labels)
    return nll,ece,brier

def TScalibrate(y_logit_cal,y_logit_test,labels_cal):
    TS_model = ModelWithTemperature()
    y_logit_cal = y_logit_cal.reshape(-1,1)
    y_logit_test = y_logit_test.reshape(-1,1)
    temperature = TS_model.get_temperature(y_logit_cal,labels_cal)
    y_logit_test_TS = y_logit_test/temperature
    return y_logit_test_TS

def PScalibrate(y_logit_cal,y_logit_test,labels_cal):
    PS_model = PlatScaling(1)
    y_logit_cal = y_logit_cal.reshape(-1,1)
    y_logit_test = y_logit_test.reshape(-1,1)
    a,b = PS_model.get_a_and_b(y_logit_cal,labels_cal)
    y_logit_test_PS = PS_model.platt_scale(y_logit_test)
    return y_logit_test_PS

def HistBinningCalibrate(y_logit_cal,y_logit_test,labels_cal,n_bins):
    y_cal = torch.sigmoid(y_logit_cal)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    # get the histogram of y_logit_cal
    hist,bin_edges = np.histogram(y_cal,bin_boundaries)
    # get the bin index of y_cal
    bin_idx = np.digitize(y_cal,bin_edges)
    
    # get the mean of labels_cal in each bin
    bin_label_mean = np.zeros(n_bins)
    for i in range(1,n_bins+1):
        if np.sum(bin_idx==i)==0:
            bin_label_mean[i-1] = (bin_lowers[i-1]+bin_uppers[i-1])/2
        else:

            bin_idx_list = [idx for idx in range(len(bin_idx)) if bin_idx[idx]==i]
            label_cpu = labels_cal[bin_idx_list]
            label_cpu = label_cpu.numpy()
            bin_label_mean[i-1] = np.mean(label_cpu)

        
    # get the bin index of y_test
    y_test = torch.sigmoid(y_logit_test)
    bin_idx_test = np.digitize(y_test,bin_boundaries)
    
    # use the mean of y_cal in each bin to calibrate y_test
    y_test_cal = np.array([bin_label_mean[i-1] for i in bin_idx_test])
    # get the calibrated y_logit_test
    y_logit_test_cal = np.log((y_test_cal+1e-10)/(1-y_test_cal+1e-10))

    
    return torch.tensor(y_logit_test_cal).to(y_logit_test.device)

# create a dataframe to store the results: name, nll before calibration, nll after calibration, ece before calibration, ece after calibration, brier before calibration, brier after calibration
calibration_method = "HistBinning"
import pandas as pd
results = pd.DataFrame(columns=["name","nll_before","nll_after","ece_before","ece_after","brier_before","brier_after"])
for idx in range(len(y_pred_cal_list)):
    name = selected_name_list[idx]
    nll_before,ece_before,brier_before = evaluate_calibration(y_pred_test_list[idx],y_test.reshape(-1,1))
    if calibration_method == "TS":
        y_pred_test_TS = TScalibrate(y_pred_cal_list[idx],y_pred_test_list[idx],y_cal.reshape(-1,1))
    elif calibration_method == "PS":
        y_pred_test_TS = PScalibrate(y_pred_cal_list[idx],y_pred_test_list[idx],y_cal.reshape(-1,1))
    elif calibration_method == "HistBinning":
        y_pred_test_TS = HistBinningCalibrate(y_pred_cal_list[idx],y_pred_test_list[idx],y_cal.reshape(-1,1),20)
    nll_after,ece_after,brier_after = evaluate_calibration(y_pred_test_TS.reshape(-1,1),y_test.reshape(-1,1))
    
    # concatenate the results
    results = pd.concat([results,pd.DataFrame({"name":[name],
                                               "nll_before":[nll_before],
                                               "nll_after":[nll_after],
                                               "ece_before":[ece_before],
                                               "ece_after":[ece_after],
                                               "brier_before":[brier_before],
                                               "brier_after":[brier_after]})])

# save the results
results.to_csv(output_dir + calibration_method + "_calibration_results.csv",index=False)




