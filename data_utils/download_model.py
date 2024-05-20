from pathlib import Path

from transformers import AutoModel, AutoTokenizer


def download_llama2(save_dir):
    model_name = "meta-llama/Llama-2-7b-hf"
    model = AutoModel.from_pretrained(
        model_name, cache_dir=str(Path(save_dir) / "Llama-2-7b-hf-local")
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=str(Path(save_dir) / "Llama-2-7b-hf-local")
    )
    pass


def download_llama3(save_dir):
    model_name = "meta-llama/Meta-Llama-3-8B"
    model = AutoModel.from_pretrained(
        model_name, cache_dir=str(Path(save_dir) / "Meta-Llama-3-8B")
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=str(Path(save_dir) / "Meta-Llama-3-8B")
    )
    pass


def download_gemma(save_dir):
    model_name = "google/gemma-7b"
    model = AutoModel.from_pretrained(
        model_name, cache_dir=str(Path(save_dir) / "gemma-7b")
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=str(Path(save_dir) / "gemma-7b")
    )
    pass


def download_deberta(save_dir):
    model_name = "microsoft/deberta-large-mnli"
    model = AutoModel.from_pretrained(
        model_name, cache_dir=str(Path(save_dir) / "deberta-large-mnli")
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=str(Path(save_dir) / "deberta-large-mnli")
    )
    pass
