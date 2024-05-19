import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from funs_feature_selection import lasso_select_k_features
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from data_utils.download_dataset import MMLU_TASKS
from utils.calibration_metrics import (
    calculate_Brier,
    calculate_ece,
    calculate_nll,
    calculate_nll_ece,
)
from utils.funs_feature_selection import (
    correlation_select_k_features,
    mutual_info_select_k_features,
)
from utils.funs_load_X_and_Y import load_MMLU_X_Y, load_X_Y
from utils.plat_scaling_self import PlatScaling
from utils.temperature_scaling_self import ModelWithTemperature

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def train_supervised_calibration(model_type, dataset_name):
    with_entropy = True
    maintain_all_entropies = True

    if model_type == "llama_2_7b":
        num_dim = 4096
        other1_num_dim = 3072  # gemma 7b
        other2_num_dim = 2048  # gemma 2b
        other1_name = "other-7B-"
        other2_name = "other-2B-"
    elif model_type == "gemma_7b":
        num_dim = 3072
        other1_num_dim = 4096  # llama2 7b
        other2_num_dim = 5120  # llama2 13b
        other1_name = "other-7B-"
        other2_name = "other-13B-"

    (
        data_train_list,
        data_test_list,
        name_list,
        y_train,
        y_test,
        SU_test,
        ask4conf_test,
    ) = load_X_Y(dataset_name, model_type, with_entropy=with_entropy)

    output_dir = "./test_output/" + dataset_name + "/" + model_type + "/"
    if with_entropy:
        output_dir += "P2_with_entropy/"
        if maintain_all_entropies:
            output_dir += "maintain_all_entropies/"
    else:
        output_dir += "P2_without_entropy/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    threshold = 0.3

    Y_train = pd.DataFrame(y_train)
    Y_train = Y_train.reset_index(drop=True)
    Y_test = pd.DataFrame(y_test)
    Y_test = Y_test.reset_index(drop=True)

    maintain_all_entropies = True
    features_from_saved_list = []
    feature_already_selected_list = []

    # include entropy features
    if maintain_all_entropies:
        for data_idx in range(len(data_train_list)):
            name = name_list[data_idx]
            # check if there are already features selected
            feature_file_name = output_dir + name + "_selected_features.json"
            if os.path.exists(feature_file_name):
                with open(feature_file_name, "r") as f:
                    feature_already_selected = json.load(f)
                feature_already_selected_list.append(feature_already_selected)
                features_from_saved_list.append(True)
                continue

            features_from_saved_list.append(False)
            if data_train_list[data_idx].shape[1] > min(
                num_dim, other1_num_dim, other2_num_dim
            ):
                if name_list[data_idx].startswith(other1_name):
                    if name_list[data_idx].startswith(
                        other1_name + "query-ans"
                    ):
                        feature_already_selected = list(
                            range(
                                2 * other1_num_dim,
                                data_train_list[data_idx].shape[1],
                            )
                        )
                    else:
                        feature_already_selected = list(
                            range(
                                other1_num_dim,
                                data_train_list[data_idx].shape[1],
                            )
                        )
                elif name_list[data_idx].startswith(other2_name):
                    if name_list[data_idx].startswith(
                        other2_name + "query-ans"
                    ):
                        feature_already_selected = list(
                            range(
                                2 * other2_num_dim,
                                data_train_list[data_idx].shape[1],
                            )
                        )
                    else:
                        feature_already_selected = list(
                            range(
                                other2_num_dim,
                                data_train_list[data_idx].shape[1],
                            )
                        )
                else:
                    if name_list[data_idx].startswith("query-ans"):
                        feature_already_selected = list(
                            range(
                                2 * num_dim, data_train_list[data_idx].shape[1]
                            )
                        )
                    else:
                        feature_already_selected = list(
                            range(num_dim, data_train_list[data_idx].shape[1])
                        )
            else:
                feature_already_selected = list(
                    range(data_train_list[data_idx].shape[1])
                )

            feature_already_selected_list.append(feature_already_selected)

    # lasso feature selection
    lasso_feature_idx_list = Parallel(n_jobs=-1, verbose=1)(
        delayed(lasso_select_k_features)(
            data_train_list[idx],
            Y_train,
            k=100,
            features_already_selected=feature_already_selected_list[idx],
            features_from_saved=features_from_saved_list[idx],
        )
        for idx in range(len(data_train_list))
    )

    for idx in range(len(data_train_list)):
        feature_already_selected_list[idx] += list(lasso_feature_idx_list[idx])

    # mutual information feature selection
    mutual_features_list = Parallel(n_jobs=-1, verbose=1)(
        delayed(mutual_info_select_k_features)(
            data_train_list[idx],
            Y_train,
            k=100,
            features_already_selected=feature_already_selected_list[idx],
            features_from_saved=features_from_saved_list[idx],
        )
        for idx in range(len(data_train_list))
    )

    for idx in range(len(data_train_list)):
        feature_already_selected_list[idx] += list(mutual_features_list[idx])

    # correlation feature selection
    correlation_features_list = Parallel(n_jobs=-1, verbose=1)(
        delayed(correlation_select_k_features)(
            data_train_list[idx],
            Y_train,
            k=100,
            features_already_selected=feature_already_selected_list[idx],
            features_from_saved=features_from_saved_list[idx],
        )
        for idx in range(len(data_train_list))
    )

    for idx in range(len(data_train_list)):
        feature_already_selected_list[idx] += list(
            correlation_features_list[idx]
        )

    for idx in range(len(feature_already_selected_list)):
        name = name_list[idx]
        save_path = output_dir + name + "_selected_features.json"
        if not features_from_saved_list[idx]:
            with open(save_path, "w") as f:
                json.dump(
                    [
                        int(feature)
                        for feature in feature_already_selected_list[idx]
                    ],
                    f,
                )

    def random_forest(X_train, Y_train, feature_already_selected, data_name):
        rf_file_name = output_dir + data_name + "_random_forest.pkl"
        if os.path.exists(rf_file_name):
            with open(rf_file_name, "rb") as f:
                reg = pickle.load(f)
            return reg
        if len(feature_already_selected) < 100:
            reg = RandomForestRegressor(
                n_estimators=100, random_state=0, max_depth=4, verbose=2
            )
        else:
            reg = RandomForestRegressor(
                n_estimators=150,
                random_state=0,
                max_depth=8,
                verbose=2,
                max_features=45,
            )

        reg.fit(X_train[:, feature_already_selected], Y_train)

        with open(rf_file_name, "wb") as f:
            pickle.dump(reg, f)
        return reg

    reg_list = Parallel(n_jobs=-1, verbose=1)(
        delayed(random_forest)(
            data_train_list[idx].detach().cpu().numpy(),
            Y_train,
            feature_already_selected_list[idx],
            name_list[idx],
        )
        for idx in range(len(data_train_list))
    )

    output_list = []

    for idx, X_test in enumerate(data_test_list):
        if X_test.shape[1] > 1:
            test_data = X_test[:, feature_already_selected_list[idx]]
            # check if there is missing value or nan in test_data
            if np.isnan(test_data).any():
                print(f"nan in {name_list[idx]}")

            output = reg_list[idx].predict(
                X_test[:, feature_already_selected_list[idx]]
            )
        else:
            output = -X_test.reshape(-1)
        output_list.append(output)

    Y_test_discrete = [1 if y > threshold else 0 for y in y_test]

    aucs = []
    fprs = []
    tprs = []
    for idx, output in enumerate(output_list):
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
    name_list.append("Semantic entropy")

    if ask4conf_test is not None:
        ask4conf_fpr, ask4conf_tpr, ask4conf_thresholds = roc_curve(
            Y_test_discrete, ask4conf_test
        )
        ask4conf_roc_auc = auc(ask4conf_fpr, ask4conf_tpr)
        fprs.append(ask4conf_fpr)
        tprs.append(ask4conf_tpr)
        aucs.append(ask4conf_roc_auc)
        name_list.append("Ask4Conf")

    plt.figure()

    for idx in range(len(aucs)):
        plt.plot(
            fprs[idx],
            tprs[idx],
            lw=1,
            label=f"{name_list[idx]}, area = {aucs[idx]:.3f}",
        )

    # plt.plot(SU_fpr, SU_tpr, color='grey',linestyle='--', lw=1, label='Semantic entropy (area = %0.3f)' % SU_roc_auc)
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    dataset_formal_name = " "
    if dataset_name == "triviaqa__train":
        dataset_formal_name = f"TriviaQA (#train: {int(len(y_train))} ; #test: {int(len(y_test))})"
    elif dataset_name == "coqa__train":
        dataset_formal_name = (
            f"CoQA (#train: {int(len(y_train))} ; #test: {int(len(y_test))})"
        )
    if with_entropy:
        title_str = (
            "ROC on test data of " + dataset_formal_name + " with entropies"
        )
    else:
        title_str = (
            "ROC on test data of " + dataset_formal_name + " without entropies"
        )
    plt.title(title_str)
    plt.legend(loc="lower right")
    plt.savefig(output_dir + "p2" + title_str + ".pdf")

    file_name = output_dir + "p2_results.csv"
    results = pd.DataFrame({"model": name_list, "auc": aucs})
    results.to_csv(file_name, index=False)


def train_supervised_calibration_mmlu(
    model_type, dataset_name="mmlu", mmlu_tasks="all"
):
    with_entropy = True

    if model_type == "llama_2_7b":
        num_dim = 4096
        other1_num_dim = 3072  # gemma 7b
        other2_num_dim = 2048  # gemma 2b
        other1_name = "other-7B-"
        other2_name = "other-2B-"
    elif model_type == "gemma_7b":
        num_dim = 3072
        other1_num_dim = 4096  # llama2 7b
        other2_num_dim = 5120  # llama2 13b
        other1_name = "other-7B-"
        other2_name = "other-13B-"

    model_name = model_type

    if dataset_name.lower() == "mmlu":
        data_train_list, name_list, y_train, _ = load_MMLU_X_Y(
            phase="test",
            model_name=model_name,
            with_entropy=with_entropy,
            MMLU_TASKS=MMLU_TASKS,
        )
        data_test_list, _, y_test, ask4conf_score = load_MMLU_X_Y(
            phase="validation",
            model_name=model_name,
            with_entropy=with_entropy,
            MMLU_TASKS=MMLU_TASKS,
        )

        output_dir = "./test_output/MMLU/" + model_name + "/" + MMLU_TASKS + "/"

    if with_entropy:
        output_dir += "P2_with_entropy/"
    else:
        output_dir += "P2_without_entropy/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for data_idx, data in enumerate(data_train_list):
        print(data.shape)
        print(data_test_list[data_idx].shape)

    Y_train = pd.DataFrame(y_train)
    Y_train = Y_train.reset_index(drop=True)
    Y_test = pd.DataFrame(y_test)
    Y_test = Y_test.reset_index(drop=True)

    maintain_all_entropies = True
    features_from_saved_list = []
    feature_already_selected_list = []
    if maintain_all_entropies:
        for data_idx in range(len(data_train_list)):
            name = name_list[data_idx]
            # check if there are already features selected
            feature_file_name = output_dir + name + "_selected_features.json"
            if os.path.exists(feature_file_name):
                with open(feature_file_name, "r") as f:
                    feature_already_selected = json.load(f)
                feature_already_selected_list.append(feature_already_selected)
                features_from_saved_list.append(True)
                continue

            features_from_saved_list.append(False)
            if data_train_list[data_idx].shape[1] > min(
                num_dim, other1_num_dim, other2_num_dim
            ):
                if name_list[data_idx].startswith(other1_name):
                    if name_list[data_idx].startswith(
                        other1_name + "query-ans"
                    ):
                        feature_already_selected = list(
                            range(
                                2 * other1_num_dim,
                                data_train_list[data_idx].shape[1],
                            )
                        )
                    else:
                        feature_already_selected = list(
                            range(
                                other1_num_dim,
                                data_train_list[data_idx].shape[1],
                            )
                        )
                elif name_list[data_idx].startswith(other2_name):
                    if name_list[data_idx].startswith(
                        other2_name + "query-ans"
                    ):
                        feature_already_selected = list(
                            range(
                                2 * other2_num_dim,
                                data_train_list[data_idx].shape[1],
                            )
                        )
                    else:
                        feature_already_selected = list(
                            range(
                                other2_num_dim,
                                data_train_list[data_idx].shape[1],
                            )
                        )
                else:
                    if name_list[data_idx].startswith("query-ans"):
                        feature_already_selected = list(
                            range(
                                2 * num_dim, data_train_list[data_idx].shape[1]
                            )
                        )
                    else:
                        feature_already_selected = list(
                            range(num_dim, data_train_list[data_idx].shape[1])
                        )
            else:
                feature_already_selected = list(
                    range(data_train_list[data_idx].shape[1])
                )

            feature_already_selected_list.append(feature_already_selected)

    lasso_feature_idx_list = Parallel(n_jobs=-1, verbose=1)(
        delayed(lasso_select_k_features)(
            data_train_list[idx],
            Y_train,
            k=100,
            features_already_selected=feature_already_selected_list[idx],
            features_from_saved=features_from_saved_list[idx],
        )
        for idx in range(len(data_train_list))
    )

    for idx in range(len(data_train_list)):
        feature_already_selected_list[idx] += list(lasso_feature_idx_list[idx])

    mutual_features_list = Parallel(n_jobs=-1, verbose=1)(
        delayed(mutual_info_select_k_features)(
            data_train_list[idx],
            Y_train,
            k=100,
            features_already_selected=feature_already_selected_list[idx],
            features_from_saved=features_from_saved_list[idx],
        )
        for idx in range(len(data_train_list))
    )

    for idx in range(len(data_train_list)):
        feature_already_selected_list[idx] += list(mutual_features_list[idx])

    correlation_features_list = Parallel(n_jobs=-1, verbose=1)(
        delayed(correlation_select_k_features)(
            data_train_list[idx],
            Y_train,
            k=100,
            features_already_selected=feature_already_selected_list[idx],
            features_from_saved=features_from_saved_list[idx],
        )
        for idx in range(len(data_train_list))
    )

    for idx in range(len(data_train_list)):
        feature_already_selected_list[idx] += list(
            correlation_features_list[idx]
        )

    for idx in range(len(feature_already_selected_list)):
        name = name_list[idx]
        save_path = output_dir + name + "_selected_features.json"
        if not features_from_saved_list[idx]:
            with open(save_path, "w") as f:
                json.dump(
                    [
                        int(feature)
                        for feature in feature_already_selected_list[idx]
                    ],
                    f,
                )

    def random_forest(X_train, Y_train, feature_already_selected, data_name):
        rf_file_name = output_dir + data_name + "_random_forest.pkl"
        if os.path.exists(rf_file_name):
            with open(rf_file_name, "rb") as f:
                reg = pickle.load(f)
            return reg
        if len(feature_already_selected) < 100:
            reg = RandomForestRegressor(
                n_estimators=100, random_state=0, max_depth=4, verbose=2
            )
        else:
            reg = RandomForestRegressor(
                n_estimators=150,
                random_state=0,
                max_depth=8,
                verbose=2,
                max_features=45,
            )

        reg.fit(X_train[:, feature_already_selected], Y_train)

        with open(rf_file_name, "wb") as f:
            pickle.dump(reg, f)
        return reg

    reg_list = Parallel(n_jobs=-1, verbose=1)(
        delayed(random_forest)(
            data_train_list[idx].detach().cpu().numpy(),
            Y_train,
            feature_already_selected_list[idx],
            name_list[idx],
        )
        for idx in range(len(data_train_list))
    )

    output_list = []
    for idx, X_test in enumerate(data_test_list):
        if X_test.shape[1] > 1:
            output = reg_list[idx].predict(
                X_test[:, feature_already_selected_list[idx]]
            )
        else:
            output = -X_test.reshape(-1)
        output_list.append(output)

    Y_test_discrete = list(y_test)

    aucs = []
    fprs = []
    tprs = []
    for idx, output in enumerate(output_list):
        fpr, tpr, _ = roc_curve(Y_test_discrete, output)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        fprs.append(fpr)
        tprs.append(tpr)

    if ask4conf_score is not None:
        ask4conf_fpr, ask4conf_tpr, _ = roc_curve(
            Y_test_discrete, ask4conf_score
        )
        ask4conf_auc = auc(ask4conf_fpr, ask4conf_tpr)
        aucs.append(ask4conf_auc)
        fprs.append(ask4conf_fpr)
        tprs.append(ask4conf_tpr)
        name_list.append("ask4conf")

    plt.figure()

    for idx in range(len(aucs)):
        plt.plot(
            fprs[idx],
            tprs[idx],
            lw=1,
            label=f"{name_list[idx]}, area = {aucs[idx]:.3f}",
        )

    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    if with_entropy:
        if dataset_name == "mmlu":
            title_str = "ROC on test data of MMLU with entropies"
        elif dataset_name == "alignment":
            title_str = "ROC on test data of alignment with entropies"
    else:
        if dataset_name == "mmlu":
            title_str = "ROC on test data of MMLU without entropies"
        elif dataset_name == "alignment":
            title_str = "ROC on test data of alignment without entropies"

    plt.title(title_str)
    plt.legend(loc="lower right")
    plt.savefig(output_dir + "p2" + title_str + ".pdf")

    filename = output_dir + "result.csv"
    df = pd.DataFrame({"name": name_list, "auc": aucs})
    df.to_csv(filename, index=False)


def test_calibration(model_type, dataset_name):
    with_entropy = True
    maintain_all_entropies = True
    pipeline_type = "P2"

    if dataset_name == "coqa":
        ood_dataset_name = "triviaqa"
    else:
        ood_dataset_name = "coqa"
    with_entropy = True

    _, data_test_list, name_list, _, y_test, SU_test, ask4conf_test = load_X_Y(
        dataset_name, model_type, with_entropy=with_entropy
    )

    print(len(data_test_list), len(name_list))

    y_test = [1 if y > 0.3 else 0 for y in y_test]
    y_test = torch.tensor(y_test)
    SU_test = torch.tensor(SU_test)
    ask4conf_test = torch.tensor(ask4conf_test)

    data_test_list.append(SU_test)
    name_list.append("Semantic entropy")

    data_test_list.append(ask4conf_test)
    name_list.append("Ask4conf")

    # data_list, name_list, y, SU_scores = load_X_Y(dataset_name,model_type,with_entropy=with_entropy)
    output_dir = "./test_output/" + dataset_name + "/" + model_type + "/"
    if with_entropy:
        output_dir += "P2_with_entropy/"
        if maintain_all_entropies:
            output_dir += "maintain_all_entropies/"
    else:
        output_dir += "P2_without_entropy/"

    feature_name_list = []
    random_forest_list = []
    selected_features = []

    for file in os.listdir(output_dir):
        if file.endswith("_selected_features.json"):
            rf_name = (
                file[: -len("_selected_features.json")] + "_random_forest.pkl"
            )
            # check if the rf exists
            if not os.path.exists(output_dir + rf_name):
                continue

            feature_name_list.append(file[: -len("_selected_features.json")])
            with open(output_dir + file) as f:
                selected_features.append(json.load(f))
            with open(output_dir + rf_name, "rb") as f:
                random_forest_list.append(pickle.load(f))

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
        selected_id_test_list[idx] = selected_id_test_list[idx][
            :, selected_features[idx]
        ]
        if len(selected_id_test_list[idx].shape) == 1:
            selected_id_test_list[idx] = selected_id_test_list[idx].reshape(
                -1, 1
            )

        # forward random forest
        if selected_id_test_list[idx].shape[1] > 1:
            y_pred = random_forest_list[idx].predict(selected_id_test_list[idx])
        else:
            y_pred = -selected_id_test_list[idx].reshape(-1)
        y_pred_list.append(y_pred)

    selected_name_list.append("Semantic entropy")
    selected_name_list.append("Ask4conf")

    y_pred_list.append(SU_test)
    y_pred_list.append(ask4conf_test)

    for idx in range(len(y_pred_list)):
        roc = roc_auc_score(y_test, y_pred_list[idx])
        if roc < 0.5:
            y_pred_list[idx] = -y_pred_list[idx]

    scaler = MinMaxScaler()
    y_pred_list_normalized = []
    for y_pred in y_pred_list:
        y_pred_normalized = scaler.fit_transform(y_pred.reshape(-1, 1)).reshape(
            -1
        )
        y_pred_list_normalized.append(y_pred_normalized)

    y_pred_logits_list = []
    for idx in range(len(y_pred_list_normalized)):
        y_pred_logits = y_pred_list_normalized[idx]
        y_pred_logits = (y_pred_logits + 1e-15) / (1 - y_pred_logits + 1e-15)
        y_pred_logits = np.log(y_pred_logits)
        y_pred_logits = torch.tensor(y_pred_logits)
        y_pred_logits_list.append(y_pred_logits)

    np.random.seed(0)
    idxs = np.arange(len(y_test))
    np.random.shuffle(idxs)
    calibration_ratio = 0.3
    cal_idxs = idxs[: int(len(idxs) * calibration_ratio)]
    test_idxs = idxs[int(len(idxs) * calibration_ratio) :]

    y_pred_cal_logits_list = []
    y_pred_test_logits_list = []
    y_cal = y_test[cal_idxs]
    y_test = y_test[test_idxs]
    for y_pred in y_pred_logits_list:
        y_pred_cal_logits_list.append(y_pred[cal_idxs])
        y_pred_test_logits_list.append(y_pred[test_idxs])

    y_pred_cal_list = []
    y_pred_test_list = []
    for y_pred in y_pred_list_normalized:
        y_pred_cal_list.append(torch.tensor(y_pred[cal_idxs]))
        y_pred_test_list.append(torch.tensor(y_pred[test_idxs]))

    def evaluate_calibration(y_logits, labels):
        nll, ece = calculate_nll_ece(y_logits, labels)
        brier = calculate_Brier(y_logits, labels)
        return nll, ece, brier

    def TScalibrate(y_logit_cal, y_logit_test, labels_cal):
        TS_model = ModelWithTemperature()
        y_logit_cal = y_logit_cal.reshape(-1, 1)
        y_logit_test = y_logit_test.reshape(-1, 1)
        temperature = TS_model.get_temperature(y_logit_cal, labels_cal)
        y_logit_test_TS = y_logit_test / temperature
        return y_logit_test_TS

    def PScalibrate(y_logit_cal, y_logit_test, labels_cal):
        PS_model = PlatScaling(1)
        y_logit_cal = y_logit_cal.reshape(-1, 1)
        y_logit_test = y_logit_test.reshape(-1, 1)
        a, b = PS_model.get_a_and_b(y_logit_cal, labels_cal)
        y_logit_test_PS = PS_model.platt_scale(y_logit_test)
        return y_logit_test_PS

    def HistBinningCalibrate(pred_cal, pred_test, labels_cal, n_bins):
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        bin_uppers[-1] = bin_uppers[-1] + 1e-6
        # get the histogram of y_logit_cal
        # hist,bin_edges = np.histogram(pred_cal,bin_boundaries)
        # get the bin index of y_cal
        bin_idx = np.digitize(pred_cal, bin_boundaries)

        # get the mean of labels_cal in each bin
        bin_label_mean = np.zeros(n_bins)
        for i in range(1, n_bins + 1):
            if np.sum(bin_idx == i) == 0:
                bin_label_mean[i - 1] = (
                    bin_lowers[i - 1] + bin_uppers[i - 1]
                ) / 2
            else:

                bin_idx_list = [
                    idx for idx in range(len(bin_idx)) if bin_idx[idx] == i
                ]
                label_cpu = labels_cal[bin_idx_list]
                label_cpu = label_cpu.numpy()
                bin_label_mean[i - 1] = np.mean(label_cpu)

        # get the bin index of y_test
        bin_idx_test = np.digitize(pred_test, bin_boundaries)

        # use the mean of y_cal in each bin to calibrate y_test
        y_test_cal = np.array([bin_label_mean[i - 1] for i in bin_idx_test])
        # get the calibrated y_logit_test
        y_logit_test_cal = np.log(
            (y_test_cal + 1e-10) / (1 - y_test_cal + 1e-10)
        )

        return torch.tensor(y_logit_test_cal).to(y_test.device)

    calibration_method = "HistBinning"
    results = pd.DataFrame(
        columns=[
            "name",
            "nll_before",
            "nll_after",
            "ece_before",
            "ece_after",
            "brier_before",
            "brier_after",
        ]
    )
    for idx in range(len(y_pred_cal_logits_list)):
        name = selected_name_list[idx]
        nll_before, ece_before, brier_before = evaluate_calibration(
            y_pred_test_logits_list[idx], y_test.reshape(-1, 1)
        )
        if calibration_method == "TS":
            y_pred_test_TS_logits = TScalibrate(
                y_pred_cal_logits_list[idx],
                y_pred_test_logits_list[idx],
                y_cal.reshape(-1, 1),
            )
        elif calibration_method == "PS":
            y_pred_test_TS_logits = PScalibrate(
                y_pred_cal_logits_list[idx],
                y_pred_test_logits_list[idx],
                y_cal.reshape(-1, 1),
            )
        elif calibration_method == "HistBinning":
            y_pred_test_TS_logits = HistBinningCalibrate(
                y_pred_cal_list[idx],
                y_pred_test_list[idx],
                y_cal.reshape(-1, 1),
                20,
            )
        nll_after, ece_after, brier_after = evaluate_calibration(
            y_pred_test_TS_logits.reshape(-1, 1), y_test.reshape(-1, 1)
        )

        # concatenate the results
        results = pd.concat(
            [
                results,
                pd.DataFrame(
                    {
                        "name": [name],
                        "nll_before": [nll_before],
                        "nll_after": [nll_after],
                        "ece_before": [ece_before],
                        "ece_after": [ece_after],
                        "brier_before": [brier_before],
                        "brier_after": [brier_after],
                    }
                ),
            ]
        )

    # save the results
    results.to_csv(
        output_dir + calibration_method + "_calibration_results.csv",
        index=False,
    )


def test_calibration_mmlu(model_type):
    with_entropy = True
    maintain_all_entropies = True
    pipeline_type = "P2"

    data_test_list, name_list, y_test, ask4conf_test = load_MMLU_X_Y(
        phase="validation",
        model_name=model_type,
        with_entropy=with_entropy,
        MMLU_TASKS="all",
    )

    output_dir = "./test_output/MMLU/" + model_type + "/all/"
    if with_entropy:
        output_dir += "P2_with_entropy/"
    else:
        output_dir += "P2_without_entropy/"

    y_test = [1 if y > 0.3 else 0 for y in y_test]
    y_test = torch.tensor(y_test)

    ask4conf_test = torch.tensor(ask4conf_test)

    data_test_list.append(ask4conf_test)
    name_list.append("Ask4conf")

    feature_name_list = []
    random_forest_list = []
    selected_features = []

    for file in os.listdir(output_dir):
        if file.endswith("_selected_features.json"):

            rf_name = (
                file[: -len("_selected_features.json")] + "_random_forest.pkl"
            )
            # check if the rf exists
            if not os.path.exists(output_dir + rf_name):
                continue

            feature_name_list.append(file[: -len("_selected_features.json")])
            with open(output_dir + file) as f:
                selected_features.append(json.load(f))
            with open(output_dir + rf_name, "rb") as f:
                random_forest_list.append(pickle.load(f))

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
        selected_id_test_list[idx] = selected_id_test_list[idx][
            :, selected_features[idx]
        ]
        if len(selected_id_test_list[idx].shape) == 1:
            selected_id_test_list[idx] = selected_id_test_list[idx].reshape(
                -1, 1
            )

        # forward random forest
        if selected_id_test_list[idx].shape[1] > 1:
            y_pred = random_forest_list[idx].predict(selected_id_test_list[idx])
        else:
            y_pred = -selected_id_test_list[idx].reshape(-1)
        y_pred_list.append(y_pred)

    selected_name_list.append("ask4conf")

    y_pred_list.append(ask4conf_test)

    for idx in range(len(y_pred_list)):
        roc = roc_auc_score(y_test, y_pred_list[idx])
        if roc < 0.5:
            y_pred_list[idx] = -y_pred_list[idx]

    scaler = MinMaxScaler()
    y_pred_list_normalized = []
    for y_pred in y_pred_list:
        y_pred_normalized = scaler.fit_transform(y_pred.reshape(-1, 1)).reshape(
            -1
        )
        y_pred_list_normalized.append(y_pred_normalized)
        print(min(y_pred_normalized), max(y_pred_normalized))

    y_pred_logits_list = []
    for idx in range(len(y_pred_list_normalized)):
        y_pred_logits = y_pred_list_normalized[idx]
        y_pred_logits = (y_pred_logits + 1e-15) / (1 - y_pred_logits + 1e-15)
        y_pred_logits = np.log(y_pred_logits)
        y_pred_logits = torch.tensor(y_pred_logits)
        y_pred_logits_list.append(y_pred_logits)

    np.random.seed(0)
    idxs = np.arange(len(y_test))
    np.random.shuffle(idxs)
    calibration_ratio = 0.3
    cal_idxs = idxs[: int(len(idxs) * calibration_ratio)]
    test_idxs = idxs[int(len(idxs) * calibration_ratio) :]
    y_pred_cal_list = []
    y_pred_test_list = []
    y_cal = y_test[cal_idxs]
    y_test = y_test[test_idxs]
    for y_pred in y_pred_logits_list:
        y_pred_cal_list.append(y_pred[cal_idxs])
        y_pred_test_list.append(y_pred[test_idxs])

    def evaluate_calibration(y_logits, labels):
        nll, ece = calculate_nll_ece(y_logits, labels)
        brier = calculate_Brier(y_logits, labels)
        return nll, ece, brier

    def TScalibrate(y_logit_cal, y_logit_test, labels_cal):
        TS_model = ModelWithTemperature()
        y_logit_cal = y_logit_cal.reshape(-1, 1)
        y_logit_test = y_logit_test.reshape(-1, 1)
        temperature = TS_model.get_temperature(y_logit_cal, labels_cal)
        y_logit_test_TS = y_logit_test / temperature
        return y_logit_test_TS

    def PScalibrate(y_logit_cal, y_logit_test, labels_cal):
        PS_model = PlatScaling(1)
        y_logit_cal = y_logit_cal.reshape(-1, 1)
        y_logit_test = y_logit_test.reshape(-1, 1)
        a, b = PS_model.get_a_and_b(y_logit_cal, labels_cal)
        y_logit_test_PS = PS_model.platt_scale(y_logit_test)
        return y_logit_test_PS

    def HistBinningCalibrate(y_logit_cal, y_logit_test, labels_cal, n_bins):
        y_cal = torch.sigmoid(y_logit_cal)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        # get the histogram of y_logit_cal
        hist, bin_edges = np.histogram(y_cal, bin_boundaries)
        # get the bin index of y_cal
        bin_idx = np.digitize(y_cal, bin_edges)

        # get the mean of labels_cal in each bin
        bin_label_mean = np.zeros(n_bins)
        for i in range(1, n_bins + 1):
            if np.sum(bin_idx == i) == 0:
                bin_label_mean[i - 1] = (
                    bin_lowers[i - 1] + bin_uppers[i - 1]
                ) / 2
            else:

                bin_idx_list = [
                    idx for idx in range(len(bin_idx)) if bin_idx[idx] == i
                ]
                label_cpu = labels_cal[bin_idx_list]
                label_cpu = label_cpu.numpy()
                bin_label_mean[i - 1] = np.mean(label_cpu)

        # get the bin index of y_test
        y_test = torch.sigmoid(y_logit_test)
        bin_idx_test = np.digitize(y_test, bin_boundaries)

        # use the mean of y_cal in each bin to calibrate y_test
        y_test_cal = np.array([bin_label_mean[i - 1] for i in bin_idx_test])
        # get the calibrated y_logit_test
        y_logit_test_cal = np.log(
            (y_test_cal + 1e-10) / (1 - y_test_cal + 1e-10)
        )

        return torch.tensor(y_logit_test_cal).to(y_logit_test.device)

    calibration_method = "HistBinning"
    results = pd.DataFrame(
        columns=[
            "name",
            "nll_before",
            "nll_after",
            "ece_before",
            "ece_after",
            "brier_before",
            "brier_after",
        ]
    )
    for idx in range(len(y_pred_cal_list)):
        name = selected_name_list[idx]
        nll_before, ece_before, brier_before = evaluate_calibration(
            y_pred_test_list[idx], y_test.reshape(-1, 1)
        )
        if calibration_method == "TS":
            y_pred_test_TS = TScalibrate(
                y_pred_cal_list[idx],
                y_pred_test_list[idx],
                y_cal.reshape(-1, 1),
            )
        elif calibration_method == "PS":
            y_pred_test_TS = PScalibrate(
                y_pred_cal_list[idx],
                y_pred_test_list[idx],
                y_cal.reshape(-1, 1),
            )
        elif calibration_method == "HistBinning":
            y_pred_test_TS = HistBinningCalibrate(
                y_pred_cal_list[idx],
                y_pred_test_list[idx],
                y_cal.reshape(-1, 1),
                20,
            )
        nll_after, ece_after, brier_after = evaluate_calibration(
            y_pred_test_TS.reshape(-1, 1), y_test.reshape(-1, 1)
        )

        # concatenate the results
        results = pd.concat(
            [
                results,
                pd.DataFrame(
                    {
                        "name": [name],
                        "nll_before": [nll_before],
                        "nll_after": [nll_after],
                        "ece_before": [ece_before],
                        "ece_after": [ece_after],
                        "brier_before": [brier_before],
                        "brier_after": [brier_after],
                    }
                ),
            ]
        )

    # save the results
    results.to_csv(
        output_dir + calibration_method + "_calibration_results.csv",
        index=False,
    )
