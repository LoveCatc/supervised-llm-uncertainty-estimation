import json
import os
import re
from copy import deepcopy
from pathlib import Path
from time import time
from typing import *

import datasets
import evaluate
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import transformers
from joblib import Parallel, delayed
from loguru import logger
from nltk.translate.bleu_score import sentence_bleu
from torch.nn import functional as F
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)

from data_utils.download_dataset import MMLU_TASKS
from utils.data_entry import (
    coqa_formatter,
    mmlu_formatter,
    triviaqa_formatter,
    wmt_formatter,
)
from utils.funs_get_feature_X import (
    get_average_hidden_states,
    get_entropy_statistics,
    get_last_token_hidden_states,
    get_prob_statistics,
)
from utils.funs_load_model import (
    load_llama2,  # ! this is a naming issue - it is used for all models, not just LLAMA-2; for compatibility we keep the name
)
from utils.generator_cls import INSIDEGenerator, MMLUGenerator


class StopWordStoppingCriteria(StoppingCriteria):
    """StopWord stopping criteria."""

    def __init__(self, tokenizer, stop_word):
        self.tokenizer = tokenizer
        self.stop_word = stop_word
        self.length = len(self.stop_word)

    def __call__(self, input_ids, *args, **kwargs) -> bool:
        cur_text = self.tokenizer.decode(input_ids[0])
        return cur_text[-self.length :] == self.stop_word


def generate_stopword_stopping_criteria(
    eos_words: list[str],
    tokenizer: transformers.AutoTokenizer,
) -> StoppingCriteriaList:
    stop_criteria = StoppingCriteriaList()
    for word in eos_words:
        stop_criteria.append(StopWordStoppingCriteria(tokenizer, word))
    return stop_criteria


def generate_query_X(
    model_type: Literal["llama_2_7b", "llama_2_13b", "gemma_7b", "gemma_2b"],
    dataset_name: Literal["triviaqa", "coqa", "wmt"],
):
    output_dir = "test_output"
    if model_type == "gemma_7b":
        model_path = "gemma-7b"
        tokenizer_path = "gemma-7b"
    elif model_type == "gemma_2b":
        model_path = "gemma-2b"
        tokenizer_path = "gemma-2b"
    elif model_type == "llama_2_7b":
        model_path = "Llama-2-7b-hf-local"
        tokenizer_path = "Llama-2-7b-hf-local"
    elif model_type == "llama_2_13b":
        model_path = "Llama-2-13b-hf-local"
        tokenizer_path = "Llama-2-13b-hf-local"
    else:
        raise NotImplementedError(f"Model {model_type} not supported")

    model_path = "models/" + model_path
    tokenizer_path = "models/" + tokenizer_path
    model, tokenizer = load_llama2(model_path, tokenizer_path)

    hidden_state_output_dir = (
        output_dir + "/" + dataset_name + "/" + model_type + "/"
    )

    PROMPT_TOKENS = "tokenized_prompt"
    Q_BEGIN = "question_token_start_idx"
    Q_END = "answer_token_start_idx"
    if dataset_name.startswith("triviaqa"):
        data = triviaqa_formatter(
            tokenizer=tokenizer, num_example=3, cache=True
        )
        data = data[dataset_name]
    elif dataset_name.startswith("coqa"):
        data = coqa_formatter(tokenizer=tokenizer, num_example=3, cache=True)
        if dataset_name.endswith("test"):
            data = data["test"]
        elif dataset_name.endswith("train"):
            data = data["train"]
    elif dataset_name.startswith("wmt"):
        data = wmt_formatter(
            tokenizer=tokenizer, num_example=3, cache=True, conv_generation=True
        )

        data = data[dataset_name]
    if dataset_name.endswith("test"):
        # truncate data to 20000
        data = list(data.select(range(min(2000, data.num_rows))))
    elif dataset_name.startswith("wmt"):
        data = list(data.select(range(min(20000, data.num_rows))))

    output_token_average_hidden_states = True
    len_of_token_hidden_states_output = 1  # if set to zero, then not used
    get_query_entropies = True  # whether to get the entropy of the output token
    get_query_probs = True
    num_queries = len(data)

    print("queries to be processed: ", num_queries)

    if model_type == "llama_2_7b":
        layer_list = [16, 32]
        num_dim = 4096
    elif model_type == "llama_2_13b":
        layer_list = [20, 40]
        num_dim = 5120
    elif model_type == "gemma_7b":
        layer_list = [14, 28]
        num_dim = 3072
    elif model_type == "gemma_2b":
        layer_list = [9, 18]
        num_dim = 2048

    num_entropy_statistics = 4

    # initialize output_tensor as num_layers x num_queries x num_dim
    if output_token_average_hidden_states:
        output_average_tensor = torch.zeros(
            (num_queries, len(layer_list), num_dim), dtype=torch.float16
        )
    if len_of_token_hidden_states_output > 0:
        output_last_token_tensor = torch.zeros(
            (
                num_queries,
                len(layer_list),
                len_of_token_hidden_states_output,
                num_dim,
            ),
            dtype=torch.float16,
        )
    if get_query_entropies:
        entropy_output_tensor = torch.zeros(
            (num_queries, num_entropy_statistics), dtype=torch.float16
        )
    if get_query_probs:
        prob_output_tensor = torch.zeros((num_queries, 6), dtype=torch.float16)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # forward and get features of the query
    for data_i, d in tqdm(enumerate(data)):

        q_begin = d[Q_BEGIN]
        q_end = d[Q_END]

        prompt_token = d[PROMPT_TOKENS][:q_end]

        # convert prompt_token to tensor
        prompt_token = torch.tensor(prompt_token).unsqueeze(0)
        prompt_token = prompt_token.to(device)

        outputs = model.forward(prompt_token, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        if not os.path.exists(hidden_state_output_dir):
            os.makedirs(hidden_state_output_dir)

        if output_token_average_hidden_states:
            output_average_tensor[data_i] = get_average_hidden_states(
                hidden_states, layer_list, q_begin, q_end, num_dim=num_dim
            )
        if len_of_token_hidden_states_output > 0:
            output_last_token_tensor[data_i] = get_last_token_hidden_states(
                hidden_states,
                layer_list,
                q_end,
                len_of_token_hidden_states_output,
                num_dim=num_dim,
            )

        if get_query_entropies:
            entropy_output_tensor[data_i, :] = get_entropy_statistics(
                outputs.logits, q_begin, q_end
            )

        if get_query_probs:
            prob_output_tensor[data_i, :] = get_prob_statistics(
                outputs.logits, prompt_token, q_begin, q_end
            )

    # save the hidden_states output
    for idx, layer_idx in enumerate(layer_list):
        if output_token_average_hidden_states:
            torch.save(
                output_average_tensor[:, idx, :],
                hidden_state_output_dir
                + "query_average_layer_"
                + str(layer_idx)
                + ".pt",
            )
        if len_of_token_hidden_states_output > 0:
            torch.save(
                output_last_token_tensor[:, idx, :, :],
                hidden_state_output_dir
                + "query_last_"
                + str(len_of_token_hidden_states_output)
                + "_token_layer_"
                + str(layer_idx)
                + ".pt",
            )

    # release the memory
    if output_token_average_hidden_states:
        del output_average_tensor
    if len_of_token_hidden_states_output > 0:
        del output_last_token_tensor

    # save the entropy output
    if get_query_entropies:
        torch.save(
            entropy_output_tensor,
            hidden_state_output_dir + "query_entropies.pt",
        )
        # release the memory
        del entropy_output_tensor

    # save the prob output
    if get_query_probs:
        torch.save(
            prob_output_tensor, hidden_state_output_dir + "query_probs.pt"
        )
        # release the memory
        del prob_output_tensor


def generate_answer_X(
    model_type: Literal["llama_2_7b", "gemma_7b"],
    dataset_name: Literal["triviaqa", "coqa", "wmt"],
):
    output_dir = "test_output"

    if model_type == "gemma_7b":
        model_path = "gemma-7b"
        tokenizer_path = "gemma-7b"
    elif model_type == "llama_2_7b":
        model_path = "Llama-2-7b-hf-local"
        tokenizer_path = "Llama-2-7b-hf-local"

    model_path = "models/" + model_path
    tokenizer_path = "models/" + tokenizer_path
    model, tokenizer = load_llama2(model_path, tokenizer_path)

    hidden_state_output_dir = (
        output_dir + "/" + dataset_name + "/" + model_type + "/"
    )

    PROMPT_TOKENS = "tokenized_prompt"

    Q_BEGIN = "question_token_start_idx"
    Q_END = "answer_token_start_idx"
    QUERY_KEY = "question_str"
    output_token_average_hidden_states = False
    len_of_token_hidden_states_output = 0  # if set to zero, then not used
    get_query_entropies = (
        False  # whether to get the entropy of the output token
    )
    get_query_probs = False

    if model_type == "llama_2_7b":
        layer_list = [16, 32]
        num_dim = 4096
    elif model_type == "gemma_7b":
        layer_list = [14, 28]
        num_dim = 3072

    num_entropy_statistics = 4

    # generate multiple answers and get the features (statistics of entropy of output logits) of answers
    dataset_extend_name = dataset_name + "_extend.json"
    dataset_extend_path = hidden_state_output_dir + "/" + dataset_extend_name

    if dataset_name.startswith("triviaqa"):
        data = triviaqa_formatter(
            tokenizer=tokenizer, num_example=3, cache=True
        )
        data = data[dataset_name]
    elif dataset_name.startswith("coqa"):
        data = coqa_formatter(tokenizer=tokenizer, num_example=3, cache=True)
        if dataset_name.endswith("test"):
            data = data["test"]
        elif dataset_name.endswith("train"):
            data = data["train"]
    elif dataset_name.startswith("wmt"):
        data = wmt_formatter(
            tokenizer=tokenizer, num_example=3, cache=True, conv_generation=True
        )

        data = data[dataset_name]
    if dataset_name.endswith("test"):
        # truncate data to 20000
        data = list(data.select(range(min(2000, data.num_rows))))
    elif dataset_name.startswith("wmt"):
        data = list(data.select(range(min(20000, data.num_rows))))

    # if the path not exists, then create the path
    if not os.path.exists(hidden_state_output_dir):
        os.makedirs(hidden_state_output_dir)

    if os.path.exists(dataset_extend_path):
        with open(dataset_extend_path, "r") as f:
            data_extend = json.load(f)

    else:
        time1 = time()
        data_extend = list(data)
        time2 = time()
        print("Time to list the data:", time2 - time1)

    ANSWERS = "generated_answers"
    ANSWER_ENTROPY_STATISTICS = "answer_entropy_statistics"

    if dataset_name.startswith("triviaqa") or dataset_name.startswith("coqa"):
        MAX_LENGTH_OF_GENERATED_SEQUENCE = 30
        eos_words = [
            "Question:",
            " Question:",
            "\n",
            "\n\n",
            "\n\n\n",
            "\n\n\n\n",
            "\n\n\n\n\n",
            "<eos>",
            "Answer:",
            " Answer:",
            "Q:",
        ]
        NUM_GENERATION_PER_PROMPT = 10
        STEP_SIZE = 500
    elif dataset_name.startswith("cnndaily"):
        MAX_LENGTH_OF_GENERATED_SEQUENCE = 200
        eos_words = [
            "<end_of_turn>",
            "end_of_turn",
            "<start_of_turn>",
            "start_of_turn",
        ]
        NUM_GENERATION_PER_PROMPT = 10
        STEP_SIZE = 20
    elif dataset_name.startswith("wmt"):
        MAX_LENGTH_OF_GENERATED_SEQUENCE = 50
        eos_words = [
            "Q:",
            "\n",
            "\n\n",
            "\n\n\n",
            "\n\n\n\n",
            "\n\n\n\n\n",
            "<eos>",
            "A:",
            "</s><s>",
        ]
        NUM_GENERATION_PER_PROMPT = 5
        STEP_SIZE = 50

    TEMPERATURE = 1.0
    TOP_P = 1.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_entropy_statistics = 4
    num_prob_statistics = 6

    with torch.no_grad():
        generator = INSIDEGenerator(
            model=model,
            tokenizer=tokenizer,
            layer_list=layer_list,
            eos_words=eos_words,
            output_token_average_hidden_states=False,
            len_of_token_hidden_states_output=0,
            get_query_entropies=False,
            get_query_probs=False,
            layer_dim=num_dim,
        )
        from_idx = 0
        to_idx = from_idx + STEP_SIZE
        find_start_point_flag = False

        # skip the processed data
        while not find_start_point_flag:
            for idx, d in enumerate(data_extend[from_idx:to_idx]):
                if (ANSWERS not in d) or len(d[ANSWERS]) == 0:
                    find_start_point_flag = True
                    break
            if not find_start_point_flag:
                from_idx = to_idx
                to_idx = min(len(data_extend), to_idx + STEP_SIZE)

        def load_saved_data(
            hidden_state_output_dir,
            from_idx,
            to_idx,
            layer_list,
            len_of_token_hidden_states_output,
            num_entropy_statistics,
            num_prob_statistics,
        ):
            # Initialize the output
            output_average_tensors = torch.zeros(
                (
                    STEP_SIZE,
                    NUM_GENERATION_PER_PROMPT,
                    len(layer_list),
                    num_dim,
                ),
                dtype=torch.float16,
            )
            output_last_token_tensors = torch.zeros(
                (
                    STEP_SIZE,
                    NUM_GENERATION_PER_PROMPT,
                    len(layer_list),
                    len_of_token_hidden_states_output,
                    num_dim,
                ),
                dtype=torch.float16,
            )
            entropy_output_tensors = torch.zeros(
                (STEP_SIZE, NUM_GENERATION_PER_PROMPT, num_entropy_statistics),
                dtype=torch.float16,
            )
            prob_output_tensors = torch.zeros(
                (STEP_SIZE, NUM_GENERATION_PER_PROMPT, num_prob_statistics),
                dtype=torch.float16,
            )

            # check if the hidden states are already generated, if so, load the hidden states
            for idx, layer_idx in enumerate(layer_list):
                if os.path.exists(
                    hidden_state_output_dir
                    + "answer_average_layer_"
                    + str(layer_list[idx])
                    + "_"
                    + str(from_idx)
                    + "_"
                    + str(to_idx)
                    + ".pt"
                ):
                    output_average_tensors[:, :, idx, :] = torch.load(
                        hidden_state_output_dir
                        + "answer_average_layer_"
                        + str(layer_idx)
                        + "_"
                        + str(from_idx)
                        + "_"
                        + str(to_idx)
                        + ".pt"
                    ).to(device)

                if os.path.exists(
                    hidden_state_output_dir
                    + "answer_last_"
                    + str(len_of_token_hidden_states_output)
                    + "_token_layer_"
                    + str(layer_list[0])
                    + "_"
                    + str(from_idx)
                    + "_"
                    + str(to_idx)
                    + ".pt"
                ):
                    output_last_token_tensors[:, :, idx, :, :] = torch.load(
                        hidden_state_output_dir
                        + "answer_last_"
                        + str(len_of_token_hidden_states_output)
                        + "_token_layer_"
                        + str(layer_idx)
                        + "_"
                        + str(from_idx)
                        + "_"
                        + str(to_idx)
                        + ".pt"
                    ).to(device)

                elif os.path.exists(
                    hidden_state_output_dir
                    + "answer_last_5_token_layer_"
                    + str(layer_list[0])
                    + "_"
                    + str(from_idx)
                    + "_"
                    + str(to_idx)
                    + ".pt"
                ):
                    print("loading data...", from_idx, to_idx)
                    output_last_token_tensors[:, :, idx, :, :] = torch.load(
                        hidden_state_output_dir
                        + "answer_last_5_token_layer_"
                        + str(layer_idx)
                        + "_"
                        + str(from_idx)
                        + "_"
                        + str(to_idx)
                        + ".pt"
                    )[:, :, -len_of_token_hidden_states_output:, :].to(device)
            if os.path.exists(
                hidden_state_output_dir
                + "answer_entropies_"
                + str(from_idx)
                + "_"
                + str(to_idx)
                + ".pt"
            ):
                entropy_output_tensors = torch.load(
                    hidden_state_output_dir
                    + "answer_entropies_"
                    + str(from_idx)
                    + "_"
                    + str(to_idx)
                    + ".pt"
                ).to(device)
            if os.path.exists(
                hidden_state_output_dir
                + "answer_probs_"
                + str(from_idx)
                + "_"
                + str(to_idx)
                + ".pt"
            ):
                prob_output_tensors = torch.load(
                    hidden_state_output_dir
                    + "answer_probs_"
                    + str(from_idx)
                    + "_"
                    + str(to_idx)
                    + ".pt"
                ).to(device)

            return (
                output_average_tensors,
                output_last_token_tensors,
                entropy_output_tensors,
                prob_output_tensors,
            )

        # load saved data
        if output_token_average_hidden_states:
            (
                output_average_tensors,
                output_last_token_tensors,
                entropy_output_tensors,
                prob_output_tensors,
            ) = load_saved_data(
                hidden_state_output_dir,
                from_idx,
                to_idx,
                layer_list,
                len_of_token_hidden_states_output,
                num_entropy_statistics,
                num_prob_statistics,
            )

        start_idx = from_idx
        for data_i in tqdm(range(start_idx, len(data_extend))):
            d = data_extend[data_i]

            # check if this data has been processed before
            if (ANSWERS in d) and len(d[ANSWERS]) > 0:
                if not output_token_average_hidden_states:
                    continue
                if (
                    torch.sum(
                        torch.abs(output_average_tensors[data_i - from_idx])
                    )
                    > 0
                    and torch.sum(
                        torch.abs(output_last_token_tensors[data_i - from_idx])
                    )
                    > 0
                    and torch.sum(
                        torch.abs(entropy_output_tensors[data_i - from_idx])
                    )
                    > 0
                ):
                    continue

            input_length = d[Q_END]
            data_extend[data_i][ANSWERS] = []
            data_extend[data_i][ANSWER_ENTROPY_STATISTICS] = [
                [] for _ in range(NUM_GENERATION_PER_PROMPT)
            ]
            prompt_tokens = [
                d[PROMPT_TOKENS][: d[Q_END]],
            ]

            for i in range(NUM_GENERATION_PER_PROMPT):

                if output_token_average_hidden_states:
                    (
                        sequence,
                        output_average_tensor,
                        output_last_token_tensor,
                        entropy_output_tensor,
                        prob_output_tensor,
                    ) = generator.generate_with_cache(
                        prompt_tokens=prompt_tokens,
                        max_gen_len=MAX_LENGTH_OF_GENERATED_SEQUENCE,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                    )

                    data_extend[data_i][ANSWERS].append(sequence)
                    data_extend[data_i][ANSWER_ENTROPY_STATISTICS][i] = (
                        entropy_output_tensor.detach()  # type: ignore
                        .cpu()
                        .numpy()
                        .tolist()
                    )
                    output_average_tensors[data_i - from_idx, i] = (
                        output_average_tensor
                    )
                    output_last_token_tensors[data_i - from_idx, i] = (
                        output_last_token_tensor
                    )
                    entropy_output_tensors[data_i - from_idx, i] = (
                        entropy_output_tensor
                    )
                    prob_output_tensors[data_i - from_idx, i] = (
                        prob_output_tensor
                    )
                else:
                    sequence = generator.generate_with_cache(
                        prompt_tokens=prompt_tokens,
                        max_gen_len=MAX_LENGTH_OF_GENERATED_SEQUENCE,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                    )  # type: ignore
                    data_extend[data_i][ANSWERS].append(sequence)

            if data_i + 1 == to_idx:

                # save the extended data
                with open(dataset_extend_path, "w") as f:
                    json.dump(data_extend, f)

                if output_token_average_hidden_states:
                    # save the hidden_states output
                    for idx, layer_idx in enumerate(layer_list):
                        torch.save(
                            output_average_tensors[:, :, idx, :],
                            hidden_state_output_dir
                            + "answer_average_layer_"
                            + str(layer_idx)
                            + "_"
                            + str(from_idx)
                            + "_"
                            + str(to_idx)
                            + ".pt",
                        )
                        torch.save(
                            output_last_token_tensors[:, :, idx, :, :],
                            hidden_state_output_dir
                            + "answer_last_"
                            + str(len_of_token_hidden_states_output)
                            + "_token_layer_"
                            + str(layer_idx)
                            + "_"
                            + str(from_idx)
                            + "_"
                            + str(to_idx)
                            + ".pt",
                        )
                    # save the entropy output
                    torch.save(
                        entropy_output_tensors,
                        hidden_state_output_dir
                        + "answer_entropies_"
                        + str(from_idx)
                        + "_"
                        + str(to_idx)
                        + ".pt",
                    )
                    # save the prob output
                    torch.save(
                        prob_output_tensors,
                        hidden_state_output_dir
                        + "answer_probs_"
                        + str(from_idx)
                        + "_"
                        + str(to_idx)
                        + ".pt",
                    )

                to_idx = min(len(data_extend), to_idx + STEP_SIZE)
                from_idx = data_i + 1

                if output_token_average_hidden_states:
                    (
                        output_average_tensors,
                        output_last_token_tensors,
                        entropy_output_tensors,
                        prob_output_tensors,
                    ) = load_saved_data(
                        hidden_state_output_dir,
                        from_idx,
                        to_idx,
                        layer_list,
                        len_of_token_hidden_states_output,
                        num_entropy_statistics,
                        num_prob_statistics,
                    )

            if dataset_name == "triviaqa__train" and data_i > 20000:
                break
            if dataset_name == "coqa__train" and data_i > 18000:
                break
            if dataset_name == "cnndaily__train" and data_i > 10000:
                break
            if dataset_name == "wmt__train" and data_i > 20000:
                break


def generate_answer_X_most(
    model_type: Literal["llama_2_7b", "gemma_7b"],
    dataset_name: Literal["triviaqa", "coqa", "wmt"],
):
    output_dir = "test_output"

    if model_type == "gemma_7b":
        model_path = "gemma-7b"
        tokenizer_path = "gemma-7b"
    elif model_type == "llama_2_7b":
        model_path = "Llama-2-7b-hf-local"
        tokenizer_path = "Llama-2-7b-hf-local"
    else:
        raise NotImplementedError(f"Model type {model_type} not supported.")

    model_path = "models/" + model_path
    tokenizer_path = "models/" + tokenizer_path
    model, tokenizer = load_llama2(model_path, tokenizer_path)

    hidden_state_output_dir = (
        output_dir + "/" + dataset_name + "/" + model_type + "/"
    )

    PROMPT_TOKENS = "tokenized_prompt"
    Q_BEGIN = "question_token_start_idx"
    Q_END = "answer_token_start_idx"
    output_token_average_hidden_states = True
    len_of_token_hidden_states_output = 1  # if set to zero, then not used
    get_query_entropies = True  # whether to get the entropy of the output token

    if model_type == "llama_2_7b":
        layer_list = [16, 32]
        num_dim = 4096
    elif model_type == "gemma_7b":
        layer_list = [14, 28]
        num_dim = 3072

    num_entropy_statistics = 4
    num_prob_statistics = 6

    # generate multiple answers and get the features (statistics of entropy of output logits) of answers
    dataset_extend_name = dataset_name + "_mextend.json"
    dataset_extend_path = hidden_state_output_dir + "/" + dataset_extend_name
    # if the path not exists, then create the path
    if not os.path.exists(dataset_extend_path):
        if not os.path.exists(hidden_state_output_dir):
            os.makedirs(hidden_state_output_dir)

        if dataset_name.startswith("wmt"):
            data = wmt_formatter(
                tokenizer=tokenizer,
                num_example=3,
                cache=True,
                conv_generation=True,
            )
            data = data[dataset_name]

        elif dataset_name.startswith("coqa"):
            data = coqa_formatter(
                tokenizer=tokenizer, num_example=3, cache=True
            )
            if dataset_name.endswith("train"):
                data = data["train"]
            elif dataset_name.endswith("test"):
                data = data["test"]

        elif dataset_name.startswith("triviaqa"):
            data = triviaqa_formatter(
                tokenizer=tokenizer, num_example=3, cache=True
            )
            data = data[dataset_name]

        if dataset_name.endswith("train"):
            data = data.select(range(min(20000, data.num_rows)))
            print(data.num_rows)
        elif dataset_name.endswith("test"):
            data = data.select(range(min(2000, data.num_rows)))

        data_extend = list(data)
        num_query = len(data_extend)

        # Initialize the output
        output_average_tensors = torch.zeros(
            (num_query, len(layer_list), num_dim), dtype=torch.float16
        )
        output_last_token_tensors = torch.zeros(
            (
                num_query,
                len(layer_list),
                len_of_token_hidden_states_output,
                num_dim,
            ),
            dtype=torch.float16,
        )
        entropy_output_tensors = torch.zeros(
            (num_query, num_entropy_statistics), dtype=torch.float16
        )
        prob_output_tensors = torch.zeros(
            (num_query, num_prob_statistics), dtype=torch.float16
        )

    else:
        with open(dataset_extend_path) as fr:
            data_extend = json.load(fr)
        num_query = len(data_extend)
        # Initialize the output
        output_average_tensors = torch.zeros(
            (num_query, len(layer_list), num_dim), dtype=torch.float16
        )
        output_last_token_tensors = torch.zeros(
            (
                num_query,
                len(layer_list),
                len_of_token_hidden_states_output,
                num_dim,
            ),
            dtype=torch.float16,
        )

        # load the saved result:
        for idx, layer_idx in enumerate(layer_list):
            output_average_tensors[:, idx, :] = torch.load(
                hidden_state_output_dir
                + "answerm_average_layer_"
                + str(layer_idx)
                + ".pt"
            )
            output_last_token_tensors[:, idx, :, :] = torch.load(
                hidden_state_output_dir
                + "answerm_last_"
                + str(len_of_token_hidden_states_output)
                + "_token_layer_"
                + str(layer_idx)
                + ".pt"
            ).reshape(num_query, len_of_token_hidden_states_output, num_dim)
        entropy_output_tensors = torch.load(
            hidden_state_output_dir + "answerm_entropies.pt"
        )
        if os.path.exists(hidden_state_output_dir + "answerm_probs.pt"):
            prob_output_tensors = torch.load(
                hidden_state_output_dir + "answerm_probs.pt"
            )
        else:
            prob_output_tensors = torch.zeros(
                (num_query, num_prob_statistics), dtype=torch.float16
            )

    print("queries to be processed: ", len(data_extend))

    MOST_ANSWER = "most_likely_answer"
    MOST_ANSWER_ENTROPY_STATISTICS = "most_likely_answer_entropy_statistics"

    if dataset_name.startswith("triviaqa") or dataset_name.startswith("coqa"):
        MAX_LENGTH_OF_GENERATED_SEQUENCE = 30
        eos_words = [
            "Question:",
            " Question:",
            "\n",
            "\n\n",
            "\n\n\n",
            "\n\n\n\n",
            "\n\n\n\n\n",
            "<eos>",
            "Answer:",
            " Answer:",
            "Q:",
        ]
        STEP_SIZE = 50
    elif dataset_name.startswith("cnndaily"):
        MAX_LENGTH_OF_GENERATED_SEQUENCE = 200
        eos_words = [
            "<end_of_turn>",
            "end_of_turn",
            "<start_of_turn>",
            "start_of_turn",
        ]
        STEP_SIZE = 20
    elif dataset_name.startswith("wmt"):
        MAX_LENGTH_OF_GENERATED_SEQUENCE = 100
        eos_words = [
            "Q:",
            " Q:",
            "\n",
            "\n\n",
            "\n\n\n",
            "\n\n\n\n",
            "\n\n\n\n\n",
            "<eos>",
            "A:",
            " A:",
            "</s><s>",
        ]
        STEP_SIZE = 50

    TOP_P = 1.0
    period_token_id = tokenizer(". ")["input_ids"][1]

    question_framing_ids = [
        [tokenizer(eos_token)["input_ids"][1]] for eos_token in eos_words
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        generator = INSIDEGenerator(
            model=model,
            tokenizer=tokenizer,
            layer_list=layer_list,
            len_of_token_hidden_states_output=len_of_token_hidden_states_output,
            eos_words=eos_words,
            layer_dim=num_dim,
        )

        # save_flag = False

        for data_i, d in tqdm(enumerate(data_extend)):

            if MOST_ANSWER in d and len(d[MOST_ANSWER]) > 0:
                if (
                    torch.sum(torch.abs(output_average_tensors[data_i])) > 0
                    and torch.sum(torch.abs(output_last_token_tensors[data_i]))
                    > 0
                    and torch.sum(torch.abs(entropy_output_tensors[data_i])) > 0
                    and torch.sum(torch.abs(prob_output_tensors[data_i])) > 0
                ):
                    continue

            input_length = d[Q_END]
            prompt_tokens = [
                d[PROMPT_TOKENS][:input_length],
            ]
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
            (
                sequence,
                output_average_tensor,
                output_last_token_tensor,
                entropy_output_tensor,
                prob_output_tensor,
            ) = generator.generate_with_cache(
                prompt_tokens=prompt_tokens,
                max_gen_len=MAX_LENGTH_OF_GENERATED_SEQUENCE,
                temperature=-1,
                top_p=TOP_P,
            )

            data_extend[data_i][MOST_ANSWER] = sequence
            data_extend[data_i][MOST_ANSWER_ENTROPY_STATISTICS] = (
                entropy_output_tensor.detach()  # type: ignore
                .cpu()
                .numpy()
                .tolist()
            )
            output_average_tensors[data_i] = output_average_tensor  # type: ignore
            output_last_token_tensors[data_i] = output_last_token_tensor  # type: ignore
            entropy_output_tensors[data_i] = entropy_output_tensor  # type: ignore
            prob_output_tensors[data_i] = prob_output_tensor  # type: ignore

            if (data_i + 1) % STEP_SIZE == 0 or data_i + 1 == num_query:
                # save the extended data with most_likely_answer
                with open(dataset_extend_path, "w") as f:
                    json.dump(data_extend, f)

                # save the hidden_states output
                for idx, layer_idx in enumerate(layer_list):
                    torch.save(
                        output_average_tensors[:, idx, :],
                        hidden_state_output_dir
                        + "answerm_average_layer_"
                        + str(layer_idx)
                        + ".pt",
                    )
                    torch.save(
                        output_last_token_tensors[:, idx, :, :],
                        hidden_state_output_dir
                        + "answerm_last_"
                        + str(len_of_token_hidden_states_output)
                        + "_token_layer_"
                        + str(layer_idx)
                        + ".pt",
                    )

                # save the entropy output
                torch.save(
                    entropy_output_tensors,
                    hidden_state_output_dir + "answerm_entropies.pt",
                )
                torch.save(
                    prob_output_tensors,
                    hidden_state_output_dir + "answerm_probs.pt",
                )


def generate_y_most_QA(model_type, dataset_name):
    data_json_path = (
        "./test_output/"
        + dataset_name
        + "/"
        + model_type
        + "/"
        + dataset_name
        + "_mextend.json"
    )
    data_extend_path = (
        "./test_output/"
        + dataset_name
        + "/"
        + model_type
        + "/"
        + dataset_name
        + "_mextend_rouge.json"
    )

    MOST_ANSWER = "most_likely_answer"
    ANSWER_REF = "answer_str"
    if not os.path.exists(data_extend_path):
        with open(data_json_path) as fr:
            data = json.load(fr)
        data_extend_rouge = deepcopy(data)
    else:
        with open(data_extend_path) as fr:
            data_extend_rouge = json.load(fr)

    rouge_type_list = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    rouge_most = ["rouge1_most", "rouge2_most", "rougeL_most", "rougeLsum_most"]
    threshould = 0.5

    rouge = evaluate.load("rouge", keep_in_memory=True)

    def calculate_rouge(d, rouge):
        generated_answer = [d[MOST_ANSWER][0][2:].lstrip()]

        reference = [d[ANSWER_REF][2:].lstrip()]

        score = rouge.compute(
            predictions=generated_answer, references=reference
        )
        for rouge_idx, rouge_type in enumerate(rouge_type_list):

            d[rouge_most[rouge_idx]] = score[rouge_type]

    for from_idx in tqdm(range(len(data_extend_rouge))):
        # len(data_extend_rouge)
        # to_idx = min(from_idx+STEP_SIZE,len(data_extend_rouge))
        # n_job = 8
        # Parallel(n_jobs=n_job, verbose=10)(delayed(calculate_rouge)(data_extend_rouge[i],rouge) for i in range(from_idx,to_idx))
        if "rouge1_most" not in data_extend_rouge[from_idx]:
            calculate_rouge(data_extend_rouge[from_idx], rouge)

        if (from_idx + 1) % 500 == 0:

            # save the data_extend_rouge
            with open(data_extend_path, "w") as fw:
                json.dump(data_extend_rouge, fw)
        if from_idx > 18000:
            # save the data_extend_rouge
            with open(data_extend_path, "w") as fw:
                json.dump(data_extend_rouge, fw)
            break


def generate_y_most_WMT(model_type, dataset_name):
    data_json_path = (
        "./test_output/"
        + dataset_name
        + "/"
        + model_type
        + "/"
        + dataset_name
        + "_mextend.json"
    )
    data_extend_path = (
        "./test_output/"
        + dataset_name
        + "/"
        + model_type
        + "/"
        + dataset_name
        + "_mextend_bleu.json"
    )
    MOST_ANSWER = "most_likely_answer"
    ANSWER_REF = "answer_str"

    if not os.path.exists(data_extend_path):
        with open(data_json_path) as fr:
            data = json.load(fr)
        data_extend_rouge = deepcopy(data)
    else:
        with open(data_extend_path) as fr:
            data_extend_rouge = json.load(fr)

    bleu = evaluate.load("bleu")
    metric = "bleu"

    def calculate_bleu(d):
        generated_answer = [
            d[MOST_ANSWER][0].lstrip(),
        ]

        reference = [d[ANSWER_REF].lstrip()]

        if generated_answer[0] == "":
            d[metric] = 0
            return 0

        score = bleu.compute(predictions=generated_answer, references=reference)
        d[metric] = score[metric]

        return score

    for from_idx in tqdm(
        range(len(data_extend_rouge))
    ):  # len(data_extend_rouge)
        # to_idx = min(from_idx+STEP_SIZE,len(data_extend_rouge))
        # n_job = 8
        # Parallel(n_jobs=n_job, verbose=10)(delayed(calculate_rouge)(data_extend_rouge[i],rouge) for i in range(from_idx,to_idx))

        # if metric not in data_extend_rouge[from_idx]:
        calculate_bleu(data_extend_rouge[from_idx])

        if (from_idx + 1) % 500 == 0:

            # save the data_extend_rouge
            with open(data_extend_path, "w") as fw:
                json.dump(data_extend_rouge, fw)
        if from_idx > 18000:
            # save the data_extend_rouge
            with open(data_extend_path, "w") as fw:
                json.dump(data_extend_rouge, fw)
            break


def generate_ask4conf(model_type, dataset_name):
    INPUT_KEY = "tokenized_prompt"
    Q_IDX_KEY = "question_token_start_idx"
    A_IDX_KEY = "answer_token_start_idx"

    ASK4CONF_TEMPLATE = (
        "A user and a model is having a conversation.\n"
        "<user>: {q}\n"
        "<model>: {a}\n\n"
        "Please provide the probability that the model's answer is correct. Give ONLY the probability between 0.0 and 1.0, no other words or explanation.\n"
        "Probability: "
    )

    output_dir = "test_output"
    if model_type == "gemma_7b":
        model_path = "gemma-7b"
        tokenizer_path = "gemma-7b"
    elif model_type == "gemma_2b":
        model_path = "gemma-2b"
        tokenizer_path = "gemma-2b"
    elif model_type == "llama_2_7b":
        model_path = "Llama-2-7b-hf-local"
        tokenizer_path = "Llama-2-7b-hf-local"
    elif model_type == "llama_2_13b":
        model_path = "Llama-2-13b-hf-local"
        tokenizer_path = "Llama-2-13b-hf-local"
    else:
        raise NotImplementedError(f"Model {model_type} not supported")
    model_path = "models/" + model_path
    tokenizer_path = "models/" + tokenizer_path
    model, tokenizer = load_llama2(model_path, tokenizer_path)
    output_dir = Path(f"test_output/ask4conf/{model_type}")

    if dataset_name.startswith("coqa"):
        ds_name = "coqa"
        dd = coqa_formatter(tokenizer)
    elif dataset_name.startswith("trivia"):
        ds_name = "triviaqa"
        dd = triviaqa_formatter(tokenizer)
    elif dataset_name.startswith("mmlu"):
        ds_name = "mmlu"
        dd = mmlu_formatter(tokenizer)
    elif dataset_name.startswith("wmt"):
        ds_name = "wmt"
        dd = wmt_formatter(tokenizer)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported")

    counter = 0

    if ds_name.startswith("coqa") or ds_name.startswith("triviaqa"):
        eos_words = [
            "Question:",
            " Question:",
            "\n",
            "\n\n",
            "\n\n\n",
            "\n\n\n\n",
            "\n\n\n\n\n",
            "<eos>",
            "Answer:",
            " Answer:",
            "Q:",
        ]
        stop_criteria = generate_stopword_stopping_criteria(
            eos_words, tokenizer
        )
        gen_config = GenerationConfig(max_new_tokens=50)
    elif ds_name.startswith("mmlu"):
        stop_criteria = generate_stopword_stopping_criteria(
            ["\n", "\n\n", "\n\n\n", "\n\n\n\n", "\n\n\n\n\n"], tokenizer
        )
        gen_config = GenerationConfig(max_new_tokens=50)
    elif ds_name.startswith("cnndaily"):
        eos_words = [
            "<end_of_turn>",
            "end_of_turn",
            "<start_of_turn>",
            "start_of_turn",
        ]
        stop_criteria = generate_stopword_stopping_criteria(
            eos_words, tokenizer
        )
        gen_config = GenerationConfig(max_new_tokens=200)

    for dd_key, ds in dd.items():
        if not dd_key.endswith("test"):
            continue

        if ds_name.startswith("mmlu"):
            if not any([_ in dd_key for _ in MMLU_TASKS]):
                continue

        if (output_dir / f"SUCCESSFUL__{ds_name}__{dd_key}").exists():
            continue

        for ditem in tqdm(ds, desc=f"Generating {ds_name} {dd_key}"):
            input_ids = ditem[INPUT_KEY][: ditem[A_IDX_KEY]]

            with torch.no_grad():
                model_answer = model.generate(
                    inputs=torch.tensor(input_ids, dtype=torch.long)
                    .reshape(1, -1)
                    .cuda(),
                    stopping_criteria=stop_criteria,
                    generation_config=gen_config,
                )

            model_answer = model_answer[0][ditem[A_IDX_KEY] :]

            ask4conf_prompt = ASK4CONF_TEMPLATE.format(
                q=tokenizer.decode(input_ids, skip_special_tokens=True).strip(),
                a=tokenizer.decode(
                    model_answer, skip_special_tokens=True
                ).strip(),
            )

            with torch.no_grad():
                tokenzied_prompt = tokenizer.encode(
                    ask4conf_prompt, return_tensors="pt"
                )
                prompt_tokens_length = tokenzied_prompt.shape[1]
                prob_answer = model.generate(
                    inputs=tokenzied_prompt.cuda(),
                    stopping_criteria=stop_criteria,
                    generation_config=GenerationConfig(max_new_tokens=10),
                )

            try:
                prob_str = re.findall(
                    r"[-+]?\d*\.\d+",
                    tokenizer.decode(prob_answer[0][prompt_tokens_length:]),
                )[0]
            except IndexError as e:
                logger.warning(
                    "Unable to find probability, could be bad generations, use 0.5. "
                )
                prob_str = 0.5

            ditem["greedy_answer_tokens"] = model_answer.tolist()
            ditem["prob_answer_tokens"] = prob_answer[0][
                prompt_tokens_length:
            ].tolist()
            ditem["prob"] = float(prob_str)

            counter += 1
            if counter >= 2500:
                break

            with open(output_dir / f"{ds_name}__{dd_key}.jsonl", "a") as fw:
                fw.write(json.dumps(ditem, ensure_ascii=False))
                fw.write("\n")

        with open(output_dir / f"SUCCESSFUL__{ds_name}__{dd_key}", "w") as fw:
            fw.write("successful")

        if counter >= 2500:
            break


def generate_uncertainty_score(model_type, dataset_name):
    ENTAILMENT_MODEL = "deberta-large-mnli"
    output_dir = "./test_output/" + dataset_name + "/" + model_type + "/"
    GENERATED_QA_LOCAL = output_dir + dataset_name + "_extend.json"
    QUESTION_KEY = "question_str"  # string
    ANSWERS_KEY = "generated_answers"  # list[list[str]]
    SEMANTIC_ENTROPY_KEY = "semantic_entropy"
    save_path = output_dir + dataset_name + "_semantic_entropy.json"

    entailment_tokenizer = AutoTokenizer.from_pretrained(
        str(Path("models") / ENTAILMENT_MODEL)
    )
    entailment_model = AutoModelForSequenceClassification.from_pretrained(
        str(Path("models") / ENTAILMENT_MODEL)
    ).cuda()

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
            if (
                ANSWERS_KEY not in data_with_answers[ridx]
                or data_with_answers[ridx][ANSWERS_KEY] == []
            ):
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
            answers = sum(row[ANSWERS_KEY], [])  # flatten the list
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
                for j in range(i + 1, len(answers_set)):
                    alist1.append(answers_set[i])
                    alist2.append(answers_set[j])

                    qa_1 = question + " " + answers[i]
                    qa_2 = question + " " + answers[j]

                    # not sure, but this seperator is used in semantic uncertainty
                    entailment_prompt = qa_1 + "[SEP]" + qa_2
                    entailment_prompts.append(entailment_prompt)

                    # here we just follow semantic uncertainty
                    encoded_prompt = entailment_tokenizer.encode(
                        entailment_prompt, padding=True
                    )
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
                        reversed_prompt, padding=True
                    )
                    reversed_pred = entailment_model(
                        # torch.tensor(
                        #     torch.tensor([encoded_reversed_prompt]),
                        #     device="cuda"
                        # )
                        torch.tensor([encoded_reversed_prompt], device="cuda")
                    )["logits"]
                    reversed_pred_label = torch.argmax(reversed_pred, dim=1)

                    if 0 in pred_label or 0 in reversed_pred_label:
                        pass  # semantically different, do nothing
                    else:  # semantically same, merge clusters
                        ans2smt[answers_set[j]] = ans2smt[answers_set[i]]

            semantic_group = list(ans2smt.values())
            group_of_answer = [ans2smt[answer] for answer in answers]
            semantic_group_set = set(semantic_group)

            # calculate the number of samples in each cluster
            num_samples_in_cluster = [
                group_of_answer.count(group_idx)
                for group_idx in semantic_group_set
            ]

            N = num_answers

            semantic_entropy = (
                -1
                / len(semantic_group_set)
                * sum(
                    [
                        np.log(num_sample / N)
                        for num_sample in num_samples_in_cluster
                    ]
                )
            )
            row[SEMANTIC_ENTROPY_KEY] = semantic_entropy

        # save the data
        if (ridx + 1) % 500 == 0:
            with open(save_path, "w") as f:
                json.dump(data_with_score, f)


def generate_query_X_mmlu(model_type, phase):
    output_dir = "test_output"
    if model_type == "gemma_7b":
        model_path = "gemma-7b"
        tokenizer_path = "gemma-7b"
    elif model_type == "llama_2_7b":
        model_path = "Llama-2-7b-hf-local"
        tokenizer_path = "Llama-2-7b-hf-local"
    elif model_type == "gemma_2b":
        model_path = "gemma-2b"
        tokenizer_path = "gemma-2b"
    elif model_type == "llama_2_13b":
        model_path = "Llama-2-13b-hf-local"
        tokenizer_path = "Llama-2-13b-hf-local"
    else:
        raise NotImplementedError(f"Model {model_type} not supported")

    model_path = "models/" + model_path
    tokenizer_path = "models/" + tokenizer_path
    model, tokenizer = load_llama2(model_path, tokenizer_path)

    if phase == "train":
        raise ValueError("The phase cannot be train")

    hidden_state_output_dir = (
        output_dir + "/MMLU/" + model_type + "/" + phase + "/"
    )

    PROMPT_TOKENS = "tokenized_prompt"
    Q_BEGIN = "question_token_start_idx"
    Q_END = "answer_token_start_idx"

    data_tasks = MMLU_TASKS

    output_token_average_hidden_states = True
    len_of_token_hidden_states_output = 1  # if set to zero, then not used
    get_query_entropies = True  # whether to get the entropy of the output token

    num_entropy_statistics = 4
    num_letters = 4

    data_total = mmlu_formatter(
        tokenizer=tokenizer,
        num_example=5,
        merge_split=False,
        conv_generation=True,
    )

    # if the path not exists, then create the path
    if not os.path.exists(hidden_state_output_dir):
        os.makedirs(hidden_state_output_dir)

    if model_type == "llama_2_7b":
        layer_list = [16, 32]
        num_dim = 4096
    elif model_type == "gemma_7b":
        layer_list = [14, 28]
        num_dim = 3072
    elif model_type == "gemma_2b":
        layer_list = [9, 18]
        num_dim = 2048
    elif model_type == "llama_2_13b":
        layer_list = [20, 40]
        num_dim = 5120

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for task in tqdm(data_tasks):
            dataset_name = "mmlu__" + task + "__" + phase
            task_output_dir = hidden_state_output_dir + task + "/"
            if not os.path.exists(task_output_dir):
                os.makedirs(task_output_dir)
            if os.path.exists(task_output_dir + "query_logits.pt"):
                continue
            data = data_total[dataset_name]

            num_queries = len(data)

            print("queries to be processed: ", num_queries)

            # initialize output_tensor as num_layers x num_queries x num_dim
            if output_token_average_hidden_states:
                output_average_tensor = torch.zeros(
                    (num_queries, len(layer_list), num_dim), dtype=torch.float16
                )
            if len_of_token_hidden_states_output > 0:
                output_last_token_tensor = torch.zeros(
                    (
                        num_queries,
                        len(layer_list),
                        len_of_token_hidden_states_output,
                        num_dim,
                    ),
                    dtype=torch.float16,
                )
            if get_query_entropies:
                entropy_output_tensor = torch.zeros(
                    (num_queries, num_entropy_statistics), dtype=torch.float16
                )

            logits_output_tensor = torch.zeros(
                (num_queries, num_letters), dtype=torch.float16
            )
            letter_tokens = [
                tokenizer.encode(letter)[1] for letter in ["A", "B", "C", "D"]
            ]

            # forward and get features of the query
            for data_i, d in tqdm(enumerate(data)):

                q_begin = d[Q_BEGIN]
                q_end = d[Q_END]
                prompt_token = d[PROMPT_TOKENS][:q_end]

                # convert prompt_token to tensor
                prompt_token = torch.tensor(prompt_token).unsqueeze(0)
                prompt_token = prompt_token.to(device)

                outputs = model.forward(prompt_token, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                logits = outputs.logits
                logits_output_tensor[data_i, :] = torch.tensor(
                    [logits[0, -1, token_idx] for token_idx in letter_tokens],
                    dtype=torch.float16,
                )

            if output_token_average_hidden_states:
                output_average_tensor[data_i] = get_average_hidden_states(
                    hidden_states, layer_list, q_begin, q_end, num_dim=num_dim
                )
            if len_of_token_hidden_states_output > 0:
                output_last_token_tensor[data_i] = get_last_token_hidden_states(
                    hidden_states,
                    layer_list,
                    q_end,
                    len_of_token_hidden_states_output,
                    num_dim=num_dim,
                )

            if get_query_entropies:
                entropy_output_tensor[data_i, :] = get_entropy_statistics(
                    outputs.logits, q_begin, q_end
                )

            # save the hidden_states output
            for idx, layer_idx in enumerate(layer_list):
                if output_token_average_hidden_states:
                    torch.save(
                        output_average_tensor[:, idx, :],
                        task_output_dir
                        + "query_average_layer_"
                        + str(layer_idx)
                        + ".pt",
                    )
                if len_of_token_hidden_states_output > 0:
                    torch.save(
                        output_last_token_tensor[:, idx, :, :],
                        task_output_dir
                        + "query_last_"
                        + str(len_of_token_hidden_states_output)
                        + "_token_layer_"
                        + str(layer_idx)
                        + ".pt",
                    )

            # release the memory
            if output_token_average_hidden_states:
                del output_average_tensor
            if len_of_token_hidden_states_output > 0:
                del output_last_token_tensor

            # save the entropy output
            if get_query_entropies:
                torch.save(
                    entropy_output_tensor,
                    task_output_dir + "query_entropies.pt",
                )
                # release the memory
                del entropy_output_tensor

            # save the logits output
            torch.save(
                logits_output_tensor, task_output_dir + "query_logits.pt"
            )


def generate_answer_X_mmlu(model_type, phase):
    output_dir = "test_output"

    if model_type == "gemma_7b":
        model_path = "gemma-7b"
        tokenizer_path = "gemma-7b"
    elif model_type == "llama_2_7b":
        model_path = "Llama-2-7b-hf-local"
        tokenizer_path = "Llama-2-7b-hf-local"
    elif model_type == "gemma_2b":
        model_path = "gemma-2b"
        tokenizer_path = "gemma-2b"
    elif model_type == "llama_2_13b":
        model_path = "Llama-2-13b-hf-local"
        tokenizer_path = "Llama-2-13b-hf-local"
    else:
        raise NotImplementedError(f"Model {model_type} not supported")

    model_path = "models/" + model_path
    tokenizer_path = "models/" + tokenizer_path
    model, tokenizer = load_llama2(model_path, tokenizer_path)

    if phase == "train":
        raise ValueError("The phase cannot be train")

    hidden_state_output_dir = (
        output_dir + "/MMLU/" + model_type + "/" + phase + "/"
    )

    PROMPT_TOKENS = "tokenized_prompt"
    Q_BEGIN = "question_token_start_idx"
    Q_END = "answer_token_start_idx"
    QUERY_KEY = "question_str"
    output_token_average_hidden_states = True
    len_of_token_hidden_states_output = 1  # if set to zero, then not used
    get_query_entropies = True  # whether to get the entropy of the output token
    STEP_SIZE = 500

    data_tasks = MMLU_TASKS

    if model_type == "llama_2_7b":
        layer_list = [16, 32]
        num_dim = 4096
    elif model_type == "gemma_7b":
        layer_list = [14, 28]
        num_dim = 3072
    elif model_type == "gemma_2b":
        layer_list = [9, 18]
        num_dim = 2048
    elif model_type == "llama_2_13b":
        layer_list = [20, 40]
        num_dim = 5120

    num_entropy_statistics = 4

    data_total = mmlu_formatter(
        tokenizer=tokenizer,
        num_example=5,
        merge_split=False,
        conv_generation=True,
    )

    # if the path not exists, then create the path
    if not os.path.exists(hidden_state_output_dir):
        os.makedirs(hidden_state_output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_entropy_statistics = 4

    with torch.no_grad():
        generator = MMLUGenerator(model, tokenizer, layer_list, num_dim)

        for task in tqdm(data_tasks):
            dataset_name = "mmlu__" + task + "__" + phase
            task_output_dir = hidden_state_output_dir + task + "/"
            if not os.path.exists(task_output_dir):
                os.makedirs(task_output_dir)
            if os.path.exists(
                task_output_dir + str(layer_list[0]) + "_output_answer_X.pt"
            ):
                continue
            data = data_total[dataset_name]
            print(len(data))
            num_tokens = 4
            output_answer_X = torch.zeros(
                (len(data), num_tokens, len(layer_list), num_dim)
            )

            data = list(data)
            for i in tqdm(range(0, len(data))):
                d = data[i]
                prompt_tokens = d[PROMPT_TOKENS][: d[Q_END]]
                output_answer_X[i] = generator.generate_single(prompt_tokens)

            # save the result

            for idx, layer_idx in enumerate(layer_list):
                torch.save(
                    output_answer_X[:, :, idx, :],
                    task_output_dir + str(layer_idx) + "_output_answer_X.pt",
                )
