import json
import os
import pickle

import pandas as pd

from sklearn.metrics import auc, roc_curve

from utils.funs_load_X_and_Y import load_MMLU_X_Y, load_X_Y


def test_transferability_mmlu(model_type,dataset_name):
    with_entropy = True

    if dataset_name == "Group1":
        ood_dataset_name = "Group2"
    else:
        ood_dataset_name = "Group1"
    with_entropy=True

    data_test_list,name_list,y_test,_ = load_MMLU_X_Y(phase='validation',model_name=model_type,with_entropy=with_entropy,MMLU_TASKS = dataset_name)


    output_dir = "./test_output/MMLU/"+model_type+"/"+dataset_name+"/"
    if with_entropy:
        output_dir += "P2_with_entropy/"
    else:
        output_dir += "P2_without_entropy/"

    ood_output_dir = "./test_output/MMLU/"+model_type+"/"+ood_dataset_name+"/"
    if with_entropy:
        ood_output_dir += "P2_with_entropy/"
    else:
        ood_output_dir += "P2_without_entropy/"

    # load all the selected features
    feature_name_list = []
    random_forest_list = []
    selected_features = []

    for file in os.listdir(ood_output_dir):
        if file.endswith("_selected_features.json"):
            
            rf_name = file[:-len("_selected_features.json")]+"_random_forest.pkl"
            # check if the rf exists
            if not os.path.exists(ood_output_dir+rf_name):
                continue

            feature_name_list.append(file[:-len("_selected_features.json")])
            with open(ood_output_dir+file) as f:
                selected_features.append(json.load(f))
            with open(ood_output_dir+rf_name,"rb") as f:
                random_forest_list.append(pickle.load(f))


    # remain only id_test_list when name_list is in feature_name_list
    selected_id_test_list = []
    selected_name_list = []
    for name in feature_name_list:
        selected_name_list.append(name)
        name_idx = name_list.index(name)
        selected_id_test_list.append(data_test_list[name_idx])


    for idx in range(len(selected_name_list)):
        print(selected_name_list[idx])
        print(selected_features[idx])
        print(selected_id_test_list[idx].shape)


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

    fprs = []
    tprs = []
    aucs = []

    for idx in range(len(selected_name_list)):
        fpr, tpr, _ = roc_curve(y_test, y_pred_list[idx])
        roc_auc = auc(fpr, tpr)
        fprs.append(fpr)
        tprs.append(tpr)
        aucs.append(roc_auc)


    df = pd.DataFrame()
    df["name"] = selected_name_list
    df["auc"] = aucs
    df.to_csv(output_dir+"ood.csv",index=False)

def test_transferability(model_type, dataset_name):
    with_entropy = True
    maintain_all_entropies = True

    if dataset_name == "coqa":
        ood_dataset_name = "triviaqa"
    else:
        ood_dataset_name = "coqa"
    with_entropy=True

    _,data_test_list,name_list,_,y_test,_,_ = load_X_Y(dataset_name,model_type,with_entropy=with_entropy)


    # data_list, name_list, y, SU_scores = load_X_Y(dataset_name,model_type,with_entropy=with_entropy)
    output_dir = "./test_output/"+dataset_name+"/"+model_type+"/"
    if with_entropy:
        output_dir += "P2_with_entropy/"
        if maintain_all_entropies:
            output_dir += "maintain_all_entropies/"
    else:
        output_dir += "P2_without_entropy/"

    ood_output_dir = "./test_output/"+ood_dataset_name+"/"+model_type+"/"
    if with_entropy:
        ood_output_dir += "P2_with_entropy/"
        if maintain_all_entropies:
            ood_output_dir += "maintain_all_entropies/"
    else:
        ood_output_dir += "P2_without_entropy/"


    # load all the selected features
    feature_name_list = []
    random_forest_list = []
    selected_features = []
    # get all the files end with "_selected_features.json" under ood_output_dir

    for file in os.listdir(ood_output_dir):
        if file.endswith("_selected_features.json"):
            
            rf_name = file[:-len("_selected_features.json")]+"_random_forest.pkl"
            # check if the rf exists
            if not os.path.exists(ood_output_dir+rf_name):
                continue

            feature_name_list.append(file[:-len("_selected_features.json")])
            with open(ood_output_dir+file) as f:
                selected_features.append(json.load(f))
            with open(ood_output_dir+rf_name,"rb") as f:
                random_forest_list.append(pickle.load(f))


    # remain only id_test_list when name_list is in feature_name_list
    selected_id_test_list = []
    selected_name_list = []
    for name in feature_name_list:
        selected_name_list.append(name)
        name_idx = name_list.index(name)
        selected_id_test_list.append(data_test_list[name_idx])


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

    fprs = []
    tprs = []
    aucs = []
    threshold = 0.3

    y_test = [1 if y > threshold else 0 for y in y_test]
    for idx in range(len(selected_name_list)):
        fpr, tpr, _ = roc_curve(y_test, y_pred_list[idx])
        roc_auc = auc(fpr, tpr)
        fprs.append(fpr)
        tprs.append(tpr)
        aucs.append(roc_auc)


    for idx in range(len(selected_name_list)):
        print(selected_name_list[idx])
        print(aucs[idx])

    df = pd.DataFrame()
    df["name"] = selected_name_list
    df["auc"] = aucs
    df.to_csv(output_dir+"ood.csv",index=False)

