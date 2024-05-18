from pathlib import Path

import click
import datasets

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


@click.group()
@click.pass_context
def run(ctx: click.Context):
    ctx.ensure_object(dict)
    ctx.obj["data_dir"] = Path("./data")
    if not ctx.obj["data_dir"].exists():
        ctx.obj["data_dir"].mkdir(exists_ok=True, parents=True)
    ctx.obj["model_dir"] = Path("./models")
    if not ctx.obj["model_dir"].exists():
        ctx.obj["model_dir"].mkdir(exists_ok=True, parents=True)
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
def generate_ds(ctx: click.Context):
    """
    Generate the dataset for supervised calibration,
    including features of white-box and grey-box,
    as well as the labels of several target scores.
    """
    pass


@run.command()
@click.pass_context
def train_model(ctx: click.Context):
    """
    Train the supervised calibration model.
    """
    pass
