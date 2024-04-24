from __future__ import annotations

from pathlib import Path
from typing import Iterator

import datasets
import pandas as pd
import transformers
from tqdm.auto import tqdm
from loguru import logger
from transformers import AutoTokenizer
import re

COQA_LOCAL = "./data/coqa"
TRIVIA_LOCAL = "./data/trivia_qa"
MMLU_LOCAL = "./data/mmlu"
CNN_LOCAL = "./data/cnn_dailymail"
WMT_LOCAL = "./data/wmt_selected"
WEBGPT_LOCAL = "./data/webgpt_comparisons"

CACHE_LOCAL = "./data_cache"

ENTER_PAT = re.compile(r"\n")


def normalize_text(text):
    # Remove space before punctuation
    text = re.sub(r"\s+([.,;?!:])", r"\1", text)

    # Fix spacing after punctuation if missing
    text = re.sub(r"([.,;?!:])([^\s])", r"\1 \2", text)

    return text


def coqa_formatter(
    tokenizer: transformers.PreTrainedTokenizer,
    dpath: str = COQA_LOCAL,
    num_example: int = 3,
    cache: bool = True,
) -> datasets.DatasetDict:
    step_size = 1 + num_example
    dd = datasets.load_from_disk(dpath)
    merged_datasets = {}

    caching_path = str(
        Path(CACHE_LOCAL) / f"coqa_{tokenizer.__class__.__name__}_exmp{num_example}"
    )

    if cache:
        if Path(caching_path).exists():
            logger.info(f"Loading cached dataset from {caching_path}")
            try:
                merged_datasetdict = datasets.load_from_disk(caching_path)
                return merged_datasetdict
            except:
                logger.warning(
                    f"Failed to load cached dataset from {caching_path}, need regeneration"
                )

    for ds_key, ds in dd.items():
        merged_datasets[ds_key] = []

        batch_cache = []
        batch_id = None

        for ditem in tqdm(ds, desc=f"Formatting {ds_key} dataset"):
            if batch_id != ditem["id"].split("_")[0]:
                for i in range(len(batch_cache) // step_size):
                    try:
                        chunk = batch_cache[i * step_size : (i + 1) * step_size]
                    except IndexError as e:
                        logger.warning(
                            f"Failed to chunk {batch_cache} with step_size {step_size}, could be too small chunk"
                        )
                        break

                    if not len(chunk) == step_size:
                        break
                    else:
                        story_str = (
                            f"Reading the passage and answer given questions accordingly.\n\nPassage:\n{chunk[0]['story']}\n\n"
                            f"Examples:\n"
                            + "\n".join(
                                [
                                    f"Q: {question}\nA: {answer}"
                                    for question, answer in zip(
                                        [_["question"] for _ in chunk[:-1]],
                                        [_["answer"]["text"] for _ in chunk[:-1]],
                                    )
                                ]
                            )
                            + "\n"
                        )

                        question_str = f"Q: {chunk[-1]['question']}\n"
                        answer_str = f"A: {chunk[-1]['answer']['text']}"

                        if tokenizer is not None:
                            story = tokenizer.encode(story_str)
                            question = tokenizer.encode(question_str)
                            answer = tokenizer.encode(answer_str)

                            if story[-1] == tokenizer.eos_token_id:
                                story = story[:-1]
                            if question[-1] == tokenizer.eos_token_id:
                                question = question[:-1]

                            if answer[0] == tokenizer.bos_token_id:
                                answer = answer[1:]
                            if question[0] == tokenizer.bos_token_id:
                                question = question[1:]

                            question_start_idx = len(story)
                            answer_start_idx = len(story) + len(question)

                            merged_datasets[ds_key].append(
                                {
                                    "tokenized_prompt": story + question + answer,
                                    "question_token_start_idx": question_start_idx,
                                    "answer_token_start_idx": answer_start_idx,
                                    "answer_str": answer_str,
                                    "question_str": question_str,
                                }
                            )

                        else:
                            logger.warning("no tokenizer offered, printing to stdout")
                            print(story_str + question_str + answer_str)

                batch_cache = []

                batch_id = ditem["id"].split("_")[0]
                batch_cache.append(ditem)
            else:
                batch_cache.append(ditem)

    merged_datasetdict = {("coqa__" + k): v for k, v in merged_datasets.items()}

    merged_datasetdict = datasets.DatasetDict(
        {
            k: datasets.Dataset.from_pandas(pd.DataFrame(v))
            for k, v in merged_datasets.items()
        }
    )

    if cache:
        merged_datasetdict.save_to_disk(caching_path)

    return merged_datasetdict


def triviaqa_formatter(
    tokenizer: transformers.PreTrainedTokenizer,
    dpath: str = TRIVIA_LOCAL,
    num_example: int = 3,
    cache: bool = True,
) -> datasets.DatasetDict:
    step_size = 1 + num_example
    dd = datasets.load_from_disk(dpath)
    merged_datasets = {}

    caching_path = str(
        Path(CACHE_LOCAL) / f"triviaqa_{tokenizer.__class__.__name__}_exmp{num_example}"
    )

    if cache:
        if Path(caching_path).exists():
            logger.info(f"Loading cached dataset from {caching_path}")
            try:
                merged_datasetdict = datasets.load_from_disk(caching_path)
                return merged_datasetdict
            except:
                logger.warning(
                    f"Failed to load cached dataset from {caching_path}, need regeneration"
                )

    for ds_key, ds in dd.items():
        merged_datasets[ds_key] = []

        chunk_cache = []
        for idx, ditem in tqdm(enumerate(ds), desc=f"Formatting {ds_key} dataset"):
            chunk_cache.append(ditem)

            if (idx + 1) % step_size == 0:
                prompt_str = (
                    f"Answer the question like following examples.\n\n"
                    + "\n".join(
                        [
                            f"Q: {_['question']}\nA: {_['answer']['value']}"
                            for _ in chunk_cache[:-1]
                        ]
                    )
                    + "\n"
                )
                question_str = f"Q: {chunk_cache[-1]['question']}\n"
                answer_str = f"A: {chunk_cache[-1]['answer']['value']}"
                if tokenizer is not None:
                    prompt = tokenizer.encode(prompt_str)
                    question = tokenizer.encode(question_str)
                    answer = tokenizer.encode(answer_str)

                    if prompt[-1] == tokenizer.eos_token_id:
                        prompt = prompt[:-1]
                    if question[-1] == tokenizer.eos_token_id:
                        question = question[:-1]

                    if answer[0] == tokenizer.bos_token_id:
                        answer = answer[1:]
                    if question[0] == tokenizer.bos_token_id:
                        question = question[1:]

                    question_start_idx = len(prompt)
                    answer_start_idx = len(prompt) + len(question)

                    merged_datasets[ds_key].append(
                        {
                            "tokenized_prompt": prompt + question + answer,
                            "question_token_start_idx": question_start_idx,
                            "answer_token_start_idx": answer_start_idx,
                            "answer_str": answer_str,
                            "question_str": question_str,
                        }
                    )

                else:
                    logger.warning("no tokenizer offered, printing to stdout")
                    print(prompt_str + question_str + answer_str)

                # finish & clean cache
                chunk_cache = []

    merged_datasets = {("triviaqa__" + k): v for k, v in merged_datasets.items()}

    merged_datasetdict = datasets.DatasetDict(
        {
            k: datasets.Dataset.from_pandas(pd.DataFrame(v))
            for k, v in merged_datasets.items()
        }
    )

    if cache:
        merged_datasetdict.save_to_disk(caching_path)

    return merged_datasetdict


def mmlu_formatter(
    tokenizer: transformers.PreTrainedTokenizer,
    dpath: str = MMLU_LOCAL,
    num_example: int = 5,
    cache: bool = True,
    merge_split: bool = False,
    conv_generation: bool = True,
) -> datasets.DatasetDict:
    step_size = 1 + num_example
    merged_datasets = {}

    caching_path = str(
        Path(CACHE_LOCAL)
        / f"mmlu_{tokenizer.__class__.__name__}_exmp{num_example}_merge{merge_split}_conv{conv_generation}"
    )

    if cache:
        if Path(caching_path).exists():
            logger.info(f"Loading cached dataset from {caching_path}")
            try:
                merged_datasetdict = datasets.load_from_disk(caching_path)
                return merged_datasetdict
            except:
                logger.warning(
                    f"Failed to load cached dataset from {caching_path}, need regeneration"
                )

    data_dirs = [
        str(Path(_)) for _ in Path(dpath).absolute().glob("*") if Path(_).is_dir()
    ]

    for f in tqdm(data_dirs, desc="Iterating over files in tar"):
        dd = datasets.load_from_disk(f)
        if merge_split:
            ds = datasets.concatenate_datasets([dd[split] for split in dd.keys()])
            ds_key = Path(f).stem
            ds_and_key = [(ds, ds_key)]
        else:
            ds_and_key = [(ds, Path(f).stem + "__" + split) for split, ds in dd.items()]

        for ds, ds_key in ds_and_key:
            if ds_key not in merged_datasets:
                merged_datasets[ds_key] = []
            chunk_cache = []
            if not conv_generation:
                for idx, row in tqdm(
                    enumerate(ds), desc=f"Formatting {ds_key} dataset", leave=False
                ):
                    chunk_cache.append(row)
                    if (idx + 1) % step_size == 0:
                        prompt_head = (
                            "You would be given a multiple-choice question paried with 4 choices (A-D). "
                            "Choose one of them using letter A, B, C, or D as the correct answer to the question. "
                            "Here are some examples: "
                        )
                        examples = "".join(
                            [
                                (
                                    f"\n\n{row['input']}"
                                    f"\nA: {row['A']}"
                                    f"\nB: {row['B']}"
                                    f"\nC: {row['C']}"
                                    f"\nD: {row['D']}"
                                    f"\n\nAnswer: {row['target']}"
                                )
                                for row in chunk_cache[:-1]
                            ]
                        )
                        examples += "\n\nNow answer the question:\n\n"
                        question = (
                            f"{chunk_cache[-1]['input']}"
                            f"\nA: {chunk_cache[-1]['A']}"
                            f"\nB: {chunk_cache[-1]['B']}"
                            f"\nC: {chunk_cache[-1]['C']}"
                            f"\nD: {chunk_cache[-1]['D']}"
                            f"\n\nAnswer: "
                        )
                        answer = f"{chunk_cache[-1]['target']}"

                        if tokenizer is not None and tokenizer.bos_token is not None:
                            prompt_head_tokens = tokenizer.encode(prompt_head)
                            examples_tokens = tokenizer.encode(examples)
                            question_tokens = tokenizer.encode(question)
                            answer_tokens = tokenizer.encode(answer)

                            if prompt_head_tokens[-1] == tokenizer.eos_token_id:
                                prompt_head_tokens = prompt_head_tokens[:-1]
                            if examples_tokens[-1] == tokenizer.eos_token_id:
                                examples_tokens = examples_tokens[:-1]
                            if question_tokens[-1] == tokenizer.eos_token_id:
                                question_tokens = question_tokens[:-1]

                            if examples_tokens[0] == tokenizer.bos_token_id:
                                examples_tokens = examples_tokens[1:]
                            if question_tokens[0] == tokenizer.bos_token_id:
                                question_tokens = question_tokens[1:]
                            if answer_tokens[0] == tokenizer.bos_token_id:
                                answer_tokens = answer_tokens[1:]

                            merged_datasets[ds_key].append(
                                {
                                    "tokenized_prompt": prompt_head_tokens
                                    + examples_tokens
                                    + question_tokens
                                    + answer_tokens,
                                    "question_token_start_idx": len(prompt_head_tokens)
                                    + len(examples_tokens),
                                    "answer_token_start_idx": len(prompt_head_tokens)
                                    + len(examples_tokens)
                                    + len(question_tokens),
                                    "answer_str": answer,
                                }
                            )

                        else:
                            merged_datasets[ds_key].append(
                                {
                                    "prompt": prompt_head + examples + question,
                                    "answer": answer,
                                }
                            )

                        chunk_cache = []

            else:
                for idx, row in tqdm(
                    enumerate(ds), desc=f"Formatting {ds_key} dataset", leave=False
                ):
                    chunk_cache.append(row)
                    if len(chunk_cache) == step_size:
                        prompt_head = (
                            "You would be given a multiple-choice question paried with 4 choices (A-D). "
                            "Choose one of them using letter A, B, C, or D as the correct answer to the question. "
                            "Here are some examples: "
                        )
                        examples = "".join(
                            [
                                (
                                    f"\n\n{row['input']}"
                                    f"\nA: {row['A']}"
                                    f"\nB: {row['B']}"
                                    f"\nC: {row['C']}"
                                    f"\nD: {row['D']}"
                                    f"\n\nAnswer: {row['target']}"
                                )
                                for row in chunk_cache[:-1]
                            ]
                        )
                        examples += "\n\nNow answer the question:\n\n"
                        question = (
                            f"{chunk_cache[-1]['input']}"
                            f"\nA: {chunk_cache[-1]['A']}"
                            f"\nB: {chunk_cache[-1]['B']}"
                            f"\nC: {chunk_cache[-1]['C']}"
                            f"\nD: {chunk_cache[-1]['D']}"
                            f"\n\nAnswer: "
                        )
                        answer = f"{chunk_cache[-1]['target']}"

                        if tokenizer is not None and tokenizer.bos_token is not None:
                            prompt_head_tokens = tokenizer.encode(prompt_head)
                            examples_tokens = tokenizer.encode(examples)
                            question_tokens = tokenizer.encode(question)
                            answer_tokens = tokenizer.encode(answer)

                            if prompt_head_tokens[-1] == tokenizer.eos_token_id:
                                prompt_head_tokens = prompt_head_tokens[:-1]
                            if examples_tokens[-1] == tokenizer.eos_token_id:
                                examples_tokens = examples_tokens[:-1]
                            if question_tokens[-1] == tokenizer.eos_token_id:
                                question_tokens = question_tokens[:-1]

                            if examples_tokens[0] == tokenizer.bos_token_id:
                                examples_tokens = examples_tokens[1:]
                            if question_tokens[0] == tokenizer.bos_token_id:
                                question_tokens = question_tokens[1:]
                            if answer_tokens[0] == tokenizer.bos_token_id:
                                answer_tokens = answer_tokens[1:]

                            merged_datasets[ds_key].append(
                                {
                                    "tokenized_prompt": prompt_head_tokens
                                    + examples_tokens
                                    + question_tokens
                                    + answer_tokens,
                                    "question_token_start_idx": len(prompt_head_tokens)
                                    + len(examples_tokens),
                                    "answer_token_start_idx": len(prompt_head_tokens)
                                    + len(examples_tokens)
                                    + len(question_tokens),
                                    "answer_str": answer,
                                }
                            )

                        else:
                            merged_datasets[ds_key].append(
                                {
                                    "prompt": prompt_head + examples + question,
                                    "answer": answer,
                                }
                            )

                        chunk_cache.pop(0)

    merged_datasets = {
        ("mmlu__" + k): v for k, v in merged_datasets.items() if len(v) > 0
    }

    merged_datasetdict = datasets.DatasetDict(
        {
            k: datasets.Dataset.from_pandas(pd.DataFrame(v))
            for k, v in merged_datasets.items()
        }
    )

    if cache:
        merged_datasetdict.save_to_disk(caching_path)

    return merged_datasetdict


def cnndaily_formatter(
    tokenizer: transformers.PreTrainedTokenizer,
    dpath: str = CNN_LOCAL,
    num_example: int = 3,
    cache: bool = True,
) -> datasets.DatasetDict:
    step_size = 1 + num_example
    dd = datasets.load_from_disk(dpath)
    merged_datasets = {}

    caching_path = str(
        Path(CACHE_LOCAL) / f"cnndaily_{tokenizer.__class__.__name__}_exmp{num_example}"
    )

    if cache:
        if Path(caching_path).exists():
            logger.info(f"Loading cached dataset from {caching_path}")
            try:
                merged_datasetdict = datasets.load_from_disk(caching_path)
                return merged_datasetdict
            except:
                logger.warning(
                    f"Failed to load cached dataset from {caching_path}, need regeneration"
                )

    for ds_key, ds in dd.items():
        merged_datasets[ds_key] = []

        chunk_cache = []
        for idx, ditem in tqdm(enumerate(ds), desc=f"Formatting {ds_key} dataset"):

            chunk_cache.append(ditem)

            if (idx + 1) % step_size == 0:
                prompt = "".join(
                    [
                        (
                            "<start_of_turn>user\n What are the highlights in this paragraph?: "
                            f"{c['article']}"
                            "<end_of_turn>\n"
                            "<start_of_turn>model\n the highlights of the paragraph: "
                            f"{ENTER_PAT.sub(' ', c['highlights'])}"
                            "<end_of_turn>\n"
                        )
                        for c in chunk_cache[:-1]
                    ]
                )
                question = (
                    "<start_of_turn>user\n What are the highlights in this paragraph?: "
                    f"{chunk_cache[-1]['article']}"
                    "<end_of_turn>\n"
                    "<start_of_turn>model\n the highlights of the paragraph: "
                )

                answer = (
                    f"{ENTER_PAT.sub(' ', chunk_cache[-1]['highlights'])}"
                    "<end_of_turn>\n"
                )

                prompt_tokens = tokenizer.encode(prompt)
                question_tokens = tokenizer.encode(question)
                answer_tokens = tokenizer.encode(answer)

                if prompt_tokens[-1] == tokenizer.eos_token_id:
                    prompt_tokens = prompt_tokens[:-1]
                if question_tokens[-1] == tokenizer.eos_token_id:
                    question_tokens = question_tokens[:-1]

                if question_tokens[0] == tokenizer.bos_token_id:
                    question_tokens = question_tokens[1:]
                if answer_tokens[0] == tokenizer.bos_token_id:
                    answer_tokens = answer_tokens[1:]

                question_start_idx = len(prompt_tokens)
                answer_start_idx = question_start_idx + len(question_tokens)

                merged_datasets[ds_key].append(
                    {
                        "tokenized_prompt": prompt_tokens
                        + question_tokens
                        + answer_tokens,
                        "question_token_start_idx": question_start_idx,
                        "answer_token_start_idx": answer_start_idx,
                    }
                )

                chunk_cache = []

    merged_datasets = {("cnndaily__" + k): v for k, v in merged_datasets.items()}

    merged_datasetdict = datasets.DatasetDict(
        {
            k: datasets.Dataset.from_pandas(pd.DataFrame(v))
            for k, v in merged_datasets.items()
        }
    )

    if cache:
        merged_datasetdict.save_to_disk(caching_path)

    return merged_datasetdict


def cnndailymail_formatter(
    path: str,
    tokenizer: transformers.PreTrainedTokenizer,
    n_shot: int = 2,
    cache: bool = False,
):
    dd = datasets.load_from_disk(path)
    step_size = 1 + n_shot
    merged_datasets = {}

    caching_path = str(Path(CACHE_LOCAL) / f"coqa_nexmp{n_shot}_{Path(path).name}")

    if cache:
        if Path(caching_path).exists():
            try:
                merged_datasetdict = datasets.load_from_disk(caching_path)
                return merged_datasetdict
            except:
                pass

    for ds_key, ds in dd.items():
        merged_datasets[ds_key] = []

        chunk_cache = []
        for idx, ditem in tqdm(enumerate(ds), desc=f"Formatting {ds_key} dataset"):

            chunk_cache.append(ditem)

            if (idx + 1) % step_size == 0:
                prompt = "".join(
                    [
                        (
                            "<start_of_turn>user\n What are the highlights in this paragraph?: "
                            f"{c['article']}"
                            "<end_of_turn>\n"
                            "<start_of_turn>model\n the highlights of the paragraph: "
                            f"{ENTER_PAT.sub(' ', c['highlights'])}"
                            "<end_of_turn>\n"
                        )
                        for c in chunk_cache[:-1]
                    ]
                )
                question = (
                    "<start_of_turn>user\n What are the highlights in this paragraph?: "
                    f"{chunk_cache[-1]['article']}"
                    "<end_of_turn>\n"
                    "<start_of_turn>model\n the highlights of the paragraph: "
                )

                answer = f"{ENTER_PAT.sub(' ', chunk_cache[-1]['highlights'])}"

                prompt_tokens = tokenizer.encode(prompt)
                question_tokens = tokenizer.encode(question)
                answer_tokens = tokenizer.encode(answer)

                question_start_idx = len(prompt_tokens)
                answer_start_idx = question_start_idx + len(question_tokens)

                merged_datasets[ds_key].append(
                    {
                        "prompt_str": prompt,
                        "question_str": question,
                        "answer_str": answer,
                        "tokenized_prompt": prompt_tokens
                        + question_tokens
                        + answer_tokens,
                        "tokenized_prompt_no_answer": prompt_tokens + question_tokens,
                        "question_token_start_idx": question_start_idx,
                        "answer_token_start_idx": answer_start_idx,
                    }
                )

                chunk_cache = []

    merged_datasets = {("cnndaily__" + k): v for k, v in merged_datasets.items()}

    merged_datasetdict = datasets.DatasetDict(
        {
            k: datasets.Dataset.from_pandas(pd.DataFrame(v))
            for k, v in merged_datasets.items()
        }
    )

    if cache:
        merged_datasetdict.save_to_disk(caching_path)

    return merged_datasetdict


def wmt_formatter(
    tokenizer: transformers.PreTrainedTokenizer,
    dpath: str = WMT_LOCAL,
    num_example: int = 3,
    cache: bool = True,
    conv_generation: bool = True,
    Q_LANG: str = "fr",
    A_LANG: str = "en",
) -> datasets.DatasetDict:
    step_size = 1 + num_example
    dd = datasets.load_from_disk(dpath)
    merged_datasets = {}

    caching_path = str(
        Path(CACHE_LOCAL)
        / f"wmt_{Q_LANG}2{A_LANG}_{tokenizer.__class__.__name__}_exmp{num_example}_conv{conv_generation}"
    )

    if cache:
        if Path(caching_path).exists():
            logger.info(f"Loading cached dataset from {caching_path}")
            try:
                merged_datasetdict = datasets.load_from_disk(caching_path)
                return merged_datasetdict
            except:
                logger.warning(
                    f"Failed to load cached dataset from {caching_path}, need regeneration"
                )

    for ds_key, ds in dd.items():
        merged_datasets[ds_key] = []
        chunk_cache = []
        if not conv_generation:
            for idx, ditem in tqdm(enumerate(ds), desc=f"Formatting {ds_key} dataset"):

                chunk_cache.append(ditem)

                if (idx + 1) % step_size == 0:
                    prompt = "".join(
                        [
                            (
                                f"Q: What is the English translation of the following sentence? {sen['translation'][Q_LANG]}\n"
                                f"A: {sen['translation'][A_LANG]}\n"
                            )
                            for sen in chunk_cache[:-1]
                        ]
                    )

                    question = f"Q: What is the English translation of the following sentence? {chunk_cache[-1]['translation'][Q_LANG]}\nA: "
                    answer = f"{chunk_cache[-1]['translation'][A_LANG]}"

                    if tokenizer is not None:
                        prompt_tokens = tokenizer.encode(prompt)
                        question_tokens = tokenizer.encode(question)
                        answer_tokens = tokenizer.encode(answer)

                        if prompt_tokens[-1] == tokenizer.eos_token_id:
                            prompt_tokens = prompt_tokens[:-1]
                        if question_tokens[-1] == tokenizer.eos_token_id:
                            question_tokens = question_tokens[:-1]

                        if question_tokens[0] == tokenizer.bos_token_id:
                            question_tokens = question_tokens[1:]
                        if answer_tokens[0] == tokenizer.bos_token_id:
                            answer_tokens = answer_tokens[1:]

                        question_start_idx = len(prompt_tokens)
                        answer_start_idx = question_start_idx + len(question_tokens)

                        merged_datasets[ds_key].append(
                            {
                                "tokenized_prompt": prompt_tokens
                                + question_tokens
                                + answer_tokens,
                                "question_token_start_idx": question_start_idx,
                                "answer_token_start_idx": answer_start_idx,
                                "answer_str": answer,
                                "question_str": question,
                            }
                        )
                    else:
                        print(prompt + question + answer)

                    chunk_cache = []
        else:
            for idx, ditem in tqdm(enumerate(ds), desc=f"Formatting {ds_key} dataset"):

                chunk_cache.append(ditem)

                if len(chunk_cache) == step_size:
                    prompt = "".join(
                        [
                            (
                                f"Q: What is the English translation of the following sentence? {sen['translation'][Q_LANG]}\n"
                                f"A: {sen['translation'][A_LANG]}\n"
                            )
                            for sen in chunk_cache[:-1]
                        ]
                    )

                    question = f"Q: What is the English translation of the following sentence? {chunk_cache[-1]['translation'][Q_LANG]}\nA: "
                    answer = f"{chunk_cache[-1]['translation'][A_LANG]}"

                    if tokenizer is not None:
                        prompt_tokens = tokenizer.encode(prompt)
                        question_tokens = tokenizer.encode(question)
                        answer_tokens = tokenizer.encode(answer)

                        if prompt_tokens[-1] == tokenizer.eos_token_id:
                            prompt_tokens = prompt_tokens[:-1]
                        if question_tokens[-1] == tokenizer.eos_token_id:
                            question_tokens = question_tokens[:-1]

                        if question_tokens[0] == tokenizer.bos_token_id:
                            question_tokens = question_tokens[1:]
                        if answer_tokens[0] == tokenizer.bos_token_id:
                            answer_tokens = answer_tokens[1:]

                        question_start_idx = len(prompt_tokens)
                        answer_start_idx = question_start_idx + len(question_tokens)

                        merged_datasets[ds_key].append(
                            {
                                "tokenized_prompt": prompt_tokens
                                + question_tokens
                                + answer_tokens,
                                "question_token_start_idx": question_start_idx,
                                "answer_token_start_idx": answer_start_idx,
                                "answer_str": answer,
                                "question_str": question,
                            }
                        )
                    else:
                        print(prompt + question + answer)

                    chunk_cache.pop(0)

    merged_datasets = {("wmt__" + k): v for k, v in merged_datasets.items()}

    merged_datasetdict = datasets.DatasetDict(
        {
            k: datasets.Dataset.from_pandas(pd.DataFrame(v))
            for k, v in merged_datasets.items()
        }
    )

    if cache:
        merged_datasetdict.save_to_disk(caching_path)

    return merged_datasetdict


def webgpt_formatter(
    tokenizer: transformers.PreTrainedTokenizer,
    dpath: str = WEBGPT_LOCAL,
    num_example: int = 3,
    cache: bool = True,
    conv_generation: bool = True,
) -> datasets.DatasetDict:
    step_size = 1 + num_example
    dd = datasets.load_from_disk(dpath)
    merged_datasets: dict = {}

    IDX_PATTERN = re.compile(r"\s*\[\d+.*\]\s*")

    caching_path = str(
        Path(CACHE_LOCAL) / f"webgpt_{tokenizer.__class__.__name__}_exmp{num_example}"
    )

    if cache:
        if Path(caching_path).exists():
            try:
                merged_datasetdict = datasets.load_from_disk(caching_path)
                return merged_datasetdict
            except:
                pass

    for ds_key, ds in dd.items():
        merged_datasets[ds_key] = []

        chunk_cache = []
        if not conv_generation:
            for idx, ditem in tqdm(enumerate(ds), desc=f"Formatting {ds_key} dataset"):
                if ditem["score_0"] == 0 and ditem["score_1"] == 0:
                    continue
                else:
                    chunk_cache.append(ditem)

                if (idx + 1) % step_size == 0:
                    prompt = "".join(
                        [
                            "You will receive a question with two options. Please select either answer A or B to indicate your preference. Here are some examples:\n\n"
                        ]
                        + [
                            f"Question: {d['question']['full_text']}\nA. {IDX_PATTERN.sub(' ', d['answer_0']).strip()}\nB. {IDX_PATTERN.sub(' ', d['answer_1']).strip()}\nAnswer: {'A' if d['score_0'] > d['score_1'] else 'B'}\n"
                            for d in chunk_cache[:-1]
                        ]
                    )
                    question = f"Question: {chunk_cache[-1]['question']['full_text']}\nA. {IDX_PATTERN.sub(' ', chunk_cache[-1]['answer_0']).strip()}\nB. {IDX_PATTERN.sub(' ', chunk_cache[-1]['answer_1']).strip()}\nAnswer: "
                    answer = f"{'A' if chunk_cache[-1]['score_0'] > chunk_cache[-1]['score_1'] else 'B'}"

                    prompt = normalize_text(prompt)

                    if tokenizer is not None:
                        prompt_tokens = tokenizer.encode(prompt)
                        question_tokens = tokenizer.encode(question)
                        answer_tokens = tokenizer.encode(answer)

                        if prompt_tokens[-1] == tokenizer.eos_token_id:
                            prompt_tokens = prompt_tokens[:-1]
                        if question_tokens[-1] == tokenizer.eos_token_id:
                            question_tokens = question_tokens[:-1]

                        if question_tokens[0] == tokenizer.bos_token_id:
                            question_tokens = question_tokens[1:]
                        if answer_tokens[0] == tokenizer.bos_token_id:
                            answer_tokens = answer_tokens[1:]

                        question_start_idx = len(prompt_tokens)
                        answer_start_idx = question_start_idx + len(question_tokens)

                        merged_datasets[ds_key].append(
                            {
                                "tokenized_prompt": prompt_tokens
                                + question_tokens
                                + answer_tokens,
                                "question_token_start_idx": question_start_idx,
                                "answer_token_start_idx": answer_start_idx,
                                "prompt_str": prompt,
                                "answer_str": answer,
                                "question_str": question,
                            }
                        )
                    else:
                        print(prompt + question + answer)
                        print("=" * 50)

                    chunk_cache = []
        else:
            for idx, ditem in tqdm(enumerate(ds), desc=f"Formatting {ds_key} dataset"):
                if ditem["score_0"] == 0 and ditem["score_1"] == 0:
                    continue
                else:
                    chunk_cache.append(ditem)

                if len(chunk_cache) == step_size:
                    prompt = "".join(
                        [
                            "You will receive a question with two options. Please select either answer A or B to indicate your preference. Here are some examples:\n\n"
                        ]
                        + [
                            f"Question: {d['question']['full_text']}\nA. {IDX_PATTERN.sub(' ', d['answer_0']).strip()}\nB. {IDX_PATTERN.sub(' ', d['answer_1']).strip()}\nAnswer: {'A' if d['score_0'] > d['score_1'] else 'B'}\n"
                            for d in chunk_cache[:-1]
                        ]
                    )
                    question = f"Question: {chunk_cache[-1]['question']['full_text']}\nA. {IDX_PATTERN.sub(' ', chunk_cache[-1]['answer_0']).strip()}\nB. {IDX_PATTERN.sub(' ', chunk_cache[-1]['answer_1']).strip()}\nAnswer: "
                    answer = f"{'A' if chunk_cache[-1]['score_0'] > chunk_cache[-1]['score_1'] else 'B'}"

                    prompt = normalize_text(prompt)

                    if tokenizer is not None:
                        prompt_tokens = tokenizer.encode(prompt)
                        question_tokens = tokenizer.encode(question)
                        answer_tokens = tokenizer.encode(answer)

                        if prompt_tokens[-1] == tokenizer.eos_token_id:
                            prompt_tokens = prompt_tokens[:-1]
                        if question_tokens[-1] == tokenizer.eos_token_id:
                            question_tokens = question_tokens[:-1]

                        if question_tokens[0] == tokenizer.bos_token_id:
                            question_tokens = question_tokens[1:]
                        if answer_tokens[0] == tokenizer.bos_token_id:
                            answer_tokens = answer_tokens[1:]

                        question_start_idx = len(prompt_tokens)
                        answer_start_idx = question_start_idx + len(question_tokens)

                        merged_datasets[ds_key].append(
                            {
                                "tokenized_prompt": prompt_tokens
                                + question_tokens
                                + answer_tokens,
                                "question_token_start_idx": question_start_idx,
                                "answer_token_start_idx": answer_start_idx,
                                "prompt_str": prompt,
                                "answer_str": answer,
                                "question_str": question,
                            }
                        )
                    else:
                        print(prompt + question + answer)
                        print("=" * 50)

                    chunk_cache.pop(0)

    merged_datasets = {("webgpt__" + k): v for k, v in merged_datasets.items()}

    merged_datasetdict = datasets.DatasetDict(
        {
            k: datasets.Dataset.from_pandas(pd.DataFrame(v))
            for k, v in merged_datasets.items()
        }
    )

    if cache:
        merged_datasetdict.save_to_disk(caching_path)

    return merged_datasetdict


if __name__ == "__main__":  # do cache generation
    pass
