from __future__ import annotations

from pathlib import Path

from tqdm.auto import tqdm
from loguru import logger

import datasets
import torch
import transformers
import re
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig
from transformers import StoppingCriteria, StoppingCriteriaList

from funs_load_model import load_llama2
from data_entry_new import (
    coqa_formatter,
    triviaqa_formatter,
    mmlu_formatter,
    cnndaily_formatter,
    wmt_formatter
)

INPUT_KEY = "tokenized_prompt"
Q_IDX_KEY = "question_token_start_idx"
A_IDX_KEY = "answer_token_start_idx"

MMLU_TASKS = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']

ASK4CONF_TEMPLATE = (
    "A user and a model is having a conversation.\n"
    "<user>: {q}\n"
    "<model>: {a}\n\n"
    "Please provide the probability that the model's answer is correct. Give ONLY the probability between 0.0 and 1.0, no other words or explanation.\n"
    "Probability: "
)

class StopWordStoppingCriteria(StoppingCriteria):
    """StopWord stopping criteria."""

    def __init__(self, tokenizer, stop_word):
        self.tokenizer = tokenizer
        self.stop_word = stop_word
        self.length = len(self.stop_word)

    def __call__(self, input_ids, *args, **kwargs) -> bool:
        cur_text = self.tokenizer.decode(input_ids[0])
        return cur_text[-self.length:] == self.stop_word

def generate_stopword_stopping_criteria(
    eos_words: list[str],
    tokenzier: transformers.AutoTokenizer,
    ) -> StoppingCriteriaList:
        stop_criteria = StoppingCriteriaList()
        for word in eos_words:
            stop_criteria.append(StopWordStoppingCriteria(tokenizer, word))
        return stop_criteria

if __name__ == "__main__":
    # follow generate_answer_X.py
    
    model_type = "gemma_7b"
    model_path = "gemma-7b"
    tokenizer_path = "gemma-7b"
    # model_type = "llama_2_7b"
    # model_path = "Llama-2-7b-hf-local"
    # tokenizer_path = "Llama-2-7b-hf-local"
    output_dir = Path(f"test_output/ask4conf/{model_type}")
    if not output_dir.exists():
        output_dir.mkdir(exist_ok=True)

    logger.info("Loading model")
    model, tokenizer = load_llama2(model_path, tokenizer_path)

    datadicts = {
        "coqa": coqa_formatter(tokenizer),
        "triviaqa": triviaqa_formatter(tokenizer),
        "mmlu": mmlu_formatter(tokenizer, conv_generation=True),
        "wmt": wmt_formatter(tokenizer)
        # "cnndaily": cnndaily_formatter(tokenizer)
    }

    logger.info("Start generating...")
    
    for ds_name, dd in datadicts.items():
        counter = 0
        
        if ds_name.startswith("coqa") or ds_name.startswith("triviaqa"):
            eos_words = ['Question:', ' Question:', '\n', '\n\n','\n\n\n','\n\n\n\n', '\n\n\n\n\n','<eos>' ,'Answer:', ' Answer:', 'Q:']
            stop_criteria = generate_stopword_stopping_criteria(eos_words, tokenizer)
            gen_config = GenerationConfig(
                max_new_tokens=50
            )
        elif ds_name.startswith("mmlu"):
            stop_criteria = generate_stopword_stopping_criteria(['\n', '\n\n','\n\n\n','\n\n\n\n', '\n\n\n\n\n'], tokenizer)
            gen_config = GenerationConfig(
                max_new_tokens=50
            )
        elif ds_name.startswith("cnndaily"):
            eos_words = ['<end_of_turn>','end_of_turn','<start_of_turn>','start_of_turn']
            stop_criteria = generate_stopword_stopping_criteria(eos_words, tokenizer)
            gen_config = GenerationConfig(
                max_new_tokens=200
            )
        
        for dd_key, ds in dd.items():
            if not dd_key.endswith("test"):
                continue
            
            if ds_name.startswith("mmlu"):
                if not any([_ in dd_key for _ in MMLU_TASKS]):
                    continue
            
            if (output_dir / f"SUCCESSFUL__{ds_name}__{dd_key}").exists():
                continue
            
            for ditem in tqdm(ds, desc=f"Generating {ds_name} {dd_key}"):
                input_ids = ditem[INPUT_KEY][:ditem[A_IDX_KEY]]
                
                with torch.no_grad():
                    model_answer = model.generate(
                        inputs=torch.tensor(input_ids, dtype=torch.long).reshape(1, -1).cuda(),
                        stopping_criteria=stop_criteria,
                        generation_config=gen_config,
                    )
                    
                model_answer = model_answer[0][ditem[A_IDX_KEY]:]
                
                ask4conf_prompt = ASK4CONF_TEMPLATE.format(q=tokenizer.decode(input_ids, skip_special_tokens=True).strip(), a=tokenizer.decode(model_answer, skip_special_tokens=True).strip())
                
                with torch.no_grad():
                    tokenzied_prompt = tokenizer.encode(ask4conf_prompt, return_tensors="pt")
                    prompt_tokens_length = tokenzied_prompt.shape[1]
                    prob_answer = model.generate(
                        inputs=tokenzied_prompt.cuda(),
                        stopping_criteria=stop_criteria,
                        generation_config=GenerationConfig(max_new_tokens=10),
                    )
                
                try:  
                    prob_str = re.findall(r"[-+]?\d*\.\d+", 
                                      tokenizer.decode(prob_answer[0][prompt_tokens_length:]))[0]
                except IndexError as e:
                    logger.warning("Unable to find probability, could be bad generations, use 0.5. ")
                    prob_str = 0.5
                
                ditem["greedy_answer_tokens"] = model_answer.tolist()
                ditem["prob_answer_tokens"] = prob_answer[0][prompt_tokens_length:].tolist()
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