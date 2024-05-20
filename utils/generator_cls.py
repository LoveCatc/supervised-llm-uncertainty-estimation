from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MMLUGenerator:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        layer_list: List[int],
        layer_dim: int = 4096,
    ):
        # model config
        self.tokenizer = tokenizer
        self.model = model
        # self.max_seq_len = model.config.max_sequence_length
        self.max_seq_len = 2  # 2048
        self.layer_list = layer_list
        self.num_dim = layer_dim
        self.pad_id = model.config.pad_token_id

        if self.pad_id is None:
            self.pad_id = tokenizer.pad_token_id
            if self.pad_id is None:
                self.pad_id = 0

    def generate_single(
        self,
        prompt_tokens: list,
    ) -> torch.Tensor:
        """
        Generate embeddings for ABCD one by one.
        """
        bsz = 1
        letter_tokens = [
            self.tokenizer.encode(letter)[1] for letter in ["A", "B", "C", "D"]
        ]
        output_last_token_tensor = torch.zeros(
            (len(letter_tokens), len(self.layer_list), self.num_dim),
            dtype=torch.float16,
            device=device,
        )

        prompt_len = len(prompt_tokens)
        if prompt_len == 1:
            prompt_len = len(prompt_tokens[0])

        for k in range(len(letter_tokens)):
            tokens = (
                torch.full((bsz, prompt_len + 1), self.pad_id).to(device).long()
            )
            tokens[0, :prompt_len] = torch.tensor(prompt_tokens).long()
            tokens[0, prompt_len] = letter_tokens[k]
            outputs = self.model.forward(
                tokens[:, : prompt_len + 1],
                use_cache=False,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states
            for idx, layer_idx in enumerate(self.layer_list):
                output_last_token_tensor[k, idx, :] = hidden_states[layer_idx][
                    :, -1, :
                ]

        return output_last_token_tensor
