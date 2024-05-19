from pathlib import Path
from typing import *

import click
import datasets
from loguru import logger

from data_utils.download_dataset import (
    prepare_coqa,
    prepare_mmlu,
    prepare_triviaqa,
    prepare_wmt,
)
from data_utils.download_model import (
    download_deberta,
    download_gemma,
    download_llama2,
    download_llama3,
)
from supervised_calibration import (
    test_calibration,
    test_calibration_mmlu,
    train_supervised_calibration,
    train_supervised_calibration_mmlu,
)
from supervised_generation import (
    generate_answer_X,
    generate_answer_X_mmlu,
    generate_answer_X_most,
    generate_ask4conf,
    generate_query_X,
    generate_query_X_mmlu,
    generate_uncertainty_score,
    generate_y_most_QA,
    generate_y_most_WMT,
)
from uncertainty_transfer import generate_cross_LLM_data, test_transferability

AVAILABLE_DATASETS = ("coqa", "triviaqa", "mmlu", "wmt")
AVAILABLE_MODELS = ("llama_2_7b", "gemma_7b")


@click.group()
@click.pass_context
def run(ctx: click.Context):
    ctx.ensure_object(dict)
    ctx.obj["data_dir"] = Path("./data")
    if not ctx.obj["data_dir"].exists():
        ctx.obj["data_dir"].mkdir(parents=True, exist_ok=True)
    ctx.obj["model_dir"] = Path("./models")
    if not ctx.obj["model_dir"].exists():
        ctx.obj["model_dir"].mkdir(parents=True, exist_ok=True)
    pass


@run.command()
@click.pass_context
def prepare_data(ctx: click.Context):
    """
    Downloads and prepares the datasets.
    """
    prepare_coqa(ctx.obj["data_dir"])
    prepare_triviaqa(ctx.obj["data_dir"])
    prepare_mmlu(ctx.obj["data_dir"])
    prepare_wmt(ctx.obj["data_dir"])


@run.command()
@click.pass_context
def prepare_model(ctx: click.Context):
    """
    Downloads and prepares the pretrained models on huggingface.
    """
    download_gemma(ctx.obj["model_dir"])
    download_llama2(ctx.obj["model_dir"])
    download_llama3(ctx.obj["model_dir"])
    download_deberta(ctx.obj["model_dir"])


@run.command()
@click.pass_context
@click.option(
    "--models", "-m", multiple=True, type=click.Choice(AVAILABLE_MODELS)
)
@click.option(
    "--ds", "-d", multiple=True, type=click.Choice(AVAILABLE_DATASETS)
)
def generate_ds(ctx: click.Context, models: Tuple[str], ds: Tuple[str]):
    """
    Generate the dataset for supervised uncertainty estimation,
    including features of white-box and grey-box,
    as well as the labels of several target scores.
    """
    for model in models:
        for dataset in ds:
            logger.info(f"Generating dataset for {model} on {dataset}.")
            try:
                # 1. generate query, distinguish mmlu with others
                if dataset == "mmlu":
                    generate_query_X_mmlu(model, "validation")
                    generate_query_X_mmlu(model, "test")
                else:
                    generate_query_X(model, dataset)  # type: ignore
                # 2. generate answer from query, distinguish mmlu with others
                if dataset == "mmlu":
                    generate_answer_X_mmlu(model, "validation")
                    generate_answer_X_mmlu(model, "test")
                else:
                    generate_answer_X(model, dataset)  # type: ignore
                    generate_answer_X_most(model, dataset)  # type: ignore
                # 3. generate y label, distinguish wmt with others
                if dataset == "wmt":
                    generate_y_most_WMT(model, dataset)
                else:
                    generate_y_most_QA(model, dataset)

                # 4. generate other features/labels
                generate_ask4conf(model, dataset)
                generate_uncertainty_score(model, dataset)

            except Exception as e:
                logger.warning(
                    f"Failed to generate query for {model} on {dataset}. Please check."
                )
                logger.warning(e)
            logger.info(
                f"Dataset for {model} on {dataset} generated. You can now run `train-supervised` command to train the supervised model."
            )


@run.command()
@click.pass_context
@click.option(
    "--models", "-m", multiple=True, type=click.Choice(AVAILABLE_MODELS)
)
@click.option(
    "--ds", "-d", multiple=True, type=click.Choice(AVAILABLE_DATASETS)
)
def train_supervised(ctx: click.Context, models: Tuple[str], ds: Tuple[str]):
    """
    Train the supervised uncertainty estimation model.
    """
    for model in models:
        for dataset in ds:
            logger.info(f"Training supervised model for {model} on {dataset}.")
            try:
                if dataset == "mmlu":
                    train_supervised_calibration_mmlu(model)
                else:
                    train_supervised_calibration(model, dataset)  # type: ignore
            except Exception as e:
                logger.warning(
                    f"Failed to train supervised model for {model} on {dataset}. Please check."
                )
                logger.warning(e)
            logger.info(
                f"Supervised model for {model} on {dataset} trained. You can now run `evaluate-supervised` command to evaluate the model."
            )


@run.command()
@click.pass_context
@click.option(
    "--models", "-m", multiple=True, type=click.Choice(AVAILABLE_MODELS)
)
@click.option(
    "--ds", "-d", multiple=True, type=click.Choice(AVAILABLE_DATASETS)
)
def evaluate_supervised(ctx: click.Context, models: Tuple[str], ds: Tuple[str]):
    """
    Evaluate the supervised uncertainty estimation model.
    """
    for model in models:
        for dataset in ds:
            logger.info(
                f"Evaluating supervised model for {model} on {dataset}."
            )
            try:
                if dataset == "mmlu":
                    test_calibration_mmlu(model)
                else:
                    test_calibration(model, dataset)  # type: ignore
            except Exception as e:
                logger.warning(
                    f"Failed to evaluate supervised model for {model} on {dataset}. Please check."
                )
                logger.warning(e)
            logger.info(
                f"Supervised model for {model} on {dataset} evaluation done."
            )


@run.command()
@click.pass_context
@click.option(
    "--models", "-m", multiple=True, type=click.Choice(AVAILABLE_MODELS)
)
@click.option(
    "--ds", "-d", multiple=True, type=click.Choice(AVAILABLE_DATASETS)
)
def prepare_crossmodel(ctx: click.Context, models: Tuple[str], ds: Tuple[str]):
    """
    Prepare the cross-model dataset for uncertainty estimation.
    """
    for model in models:
        for dataset in ds:
            logger.info(
                f"Preparing cross-model dataset for {model} on {dataset}."
            )
            try:
                generate_cross_LLM_data(model, dataset)
            except Exception as e:
                logger.warning(
                    f"Failed to prepare cross-model dataset for {model} on {dataset}. Please check."
                )
                logger.warning(e)
            logger.info(
                f"Cross-model dataset for {model} on {dataset} prepared."
            )


@run.command()
@click.pass_context
@click.option(
    "--models", "-m", multiple=True, type=click.Choice(AVAILABLE_MODELS)
)
@click.option(
    "--ds", "-d", multiple=True, type=click.Choice(AVAILABLE_DATASETS)
)
def eval_crossmodel(ctx: click.Context, models: Tuple[str], ds: Tuple[str]):
    """
    Evaluate the transferability of the supervised uncertainty estimation method.
    """
    for model in models:
        for dataset in ds:
            logger.info(f"Evaluating transferability of {model} on {dataset}.")
            try:
                test_transferability(model, dataset)
            except Exception as e:
                logger.warning(
                    f"Failed to evaluate transferability of {model} on {dataset}. Please check."
                )
                logger.warning(e)
            logger.info(f"Transferability of {model} on {dataset} evaluated.")


if __name__ == "__main__":
    run()
