from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

def load_gemma(model_path,tokenizer_path):
    """
    Load the gemma model and tokenizer.
    """
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16
    ).cuda()
    
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False
    

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer

def load_llama2(model_path,tokenizer_path,access_token=None):
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
            torch_dtype=torch.float16
        )
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, token=access_token)
    
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False
    

    
    return model, tokenizer