import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_gemma(model_path, tokenizer_path):
    """
    Load the gemma model and tokenizer.
    """
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16
    ).cuda()

    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer


def load_llama2(model_path, tokenizer_path, access_token=None):
    """
    Load the llama2 model and tokenizer.
    """
    # Load the model
    if access_token is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            # max_memory = {0:"45GB", 1:"45GB"}
        )
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            token=access_token,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, token=access_token
        )

    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # we have to handle the case where the tokenizer has an eos but not a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer
