from pprint import pprint
from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .funs_get_feature_X import (
    get_average_hidden_states,
    get_entropy_statistics,
    get_last_token_hidden_states,
)

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
    ) -> List[str]:
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


class INSIDEGenerator:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        layer_list: List[int],
        eos_words: List[str],
        output_token_average_hidden_states: bool = True,
        len_of_token_hidden_states_output: int = 5,
        get_query_entropies: bool = True,
        get_query_probs: bool = True,
        layer_dim: int = 4096,
        max_entropy_idx: int = 0,
        min_entropy_idx: int = 1,
        mean_entropy_idx: int = 2,
        std_entropy_idx: int = 3,
    ):
        # model config
        self.tokenizer = tokenizer
        self.model = model
        # self.max_seq_len = model.config.max_sequence_length
        self.max_seq_len = 8192  # 2048
        self.pad_id = model.config.pad_token_id
        self.eos_id = model.config.eos_token_id
        self.layer_list = layer_list
        self.num_dim = layer_dim
        self.output_token_average_hidden_states = (
            output_token_average_hidden_states
        )
        self.len_of_token_hidden_states_output = (
            len_of_token_hidden_states_output
        )
        self.get_query_entropies = get_query_entropies
        self.get_query_probs = get_query_probs
        self.max_entropy_idx = max_entropy_idx
        self.min_entropy_idx = min_entropy_idx
        self.mean_entropy_idx = mean_entropy_idx
        self.std_entropy_idx = std_entropy_idx
        self.eos_words = eos_words
        self.eos_tokens = [
            tokenizer.encode(eos_word)[1] for eos_word in eos_words
        ]
        self.number_words = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
        self.number_tokens = [
            tokenizer.encode(number_word)[1]
            for number_word in self.number_words
        ]

        if self.pad_id is None:
            self.pad_id = tokenizer.pad_token_id
            if self.pad_id is None:
                self.pad_id = 0
        if self.eos_id is None:
            self.eos_id = tokenizer.eos_token_id

    @torch.no_grad()
    def generate_with_cache(
        self,
        prompt_tokens: list,
        max_gen_len: int,
        temperature: float = 1.0,
        top_p: float = 0.95,
        given_answer: bool = False,
        answer_tokens: list = None,
    ) -> List[str]:
        """
        Generate text from prompts.
        Adapted from https://github.com/facebookresearch/llama/
        """

        bsz = len(prompt_tokens)
        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])
        total_len = min(self.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.pad_id).to(device).long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.pad_id

        # initialize output_tensor as num_layers x num_queries x num_dim
        if self.output_token_average_hidden_states:
            output_average_tensor = torch.zeros(
                (bsz, len(self.layer_list), self.num_dim),
                dtype=torch.float16,
                device=device,
            )
        if self.len_of_token_hidden_states_output > 0:
            output_last_token_tensor = torch.zeros(
                (
                    bsz,
                    len(self.layer_list),
                    self.len_of_token_hidden_states_output,
                    self.num_dim,
                ),
                dtype=torch.float16,
                device=device,
            )
        if self.get_query_entropies:
            entropy_list = [[] for _ in range(bsz)]
            entropy_output_tensor = torch.zeros(
                (bsz, 4), dtype=torch.float16, device=device
            )
        if self.get_query_probs:
            prob_list = [[] for _ in range(bsz)]
            prob_output_tensor = torch.zeros(
                (bsz, 6), dtype=torch.float16, device=device
            )

        start_pos = min_prompt_size
        prev_pos = 0
        answer_begin_pos = torch.tensor([len(t) for t in prompt_tokens]).to(
            device
        )
        answer_end_pos = torch.tensor(
            [len(t) + max_gen_len for t in prompt_tokens]
        ).to(device)

        for cur_pos in range(start_pos, total_len):
            if answer_end_pos.max() <= cur_pos:
                break
            outputs = self.model.forward(
                tokens[:, prev_pos:cur_pos],
                use_cache=True,
                past_key_values=(
                    outputs.past_key_values if prev_pos > 0 else None
                ),
                output_hidden_states=True,
            )

            prev_pos = cur_pos

            if cur_pos > start_pos:
                last_toks = tokens[:, cur_pos - 1]

            if not given_answer:
                next_toks = self.sample_next(
                    outputs.logits[:, -1, :], temperature, top_p
                )
            else:
                if (
                    answer_tokens is not None
                    and len(answer_tokens) > 0
                    and cur_pos < start_pos + len(answer_tokens)
                ):
                    next_toks = torch.tensor(
                        [answer_tokens[cur_pos - start_pos]]
                    ).to(device)
                elif (
                    answer_tokens is not None
                    and cur_pos >= len(answer_tokens) + start_pos
                ):
                    next_toks = self.sample_next(
                        outputs.logits[:, -1, :], temperature, top_p
                    )
                else:
                    next_toks = self.sample_next(
                        outputs.logits[:, -1, :], temperature, top_p
                    )

            # if the answer is done, then calculate the hidden_states of corresponding layers and entropy statistics
            for data_i in range(bsz):
                if (
                    (not given_answer)
                    and (cur_pos > start_pos)
                    and (last_toks[data_i] == next_toks[data_i])
                    and (
                        self.tokenizer.decode(next_toks[data_i])
                        not in self.number_words
                    )
                ):
                    answer_end_pos[data_i] = cur_pos
                if given_answer:
                    if (answer_tokens is not None) and cur_pos >= len(
                        answer_tokens
                    ) + start_pos:
                        answer_end_pos[data_i] = cur_pos
                    elif answer_tokens is None:
                        answer_end_pos[data_i] = cur_pos

                if (
                    next_toks[data_i] in self.eos_tokens
                    or self.tokenizer.decode(next_toks[data_i])
                    in self.eos_words
                ) and answer_end_pos[data_i] == len(
                    prompt_tokens[data_i]
                ) + max_gen_len:

                    answer_end_pos[data_i] = cur_pos
                if (
                    answer_begin_pos[data_i] <= cur_pos
                    and cur_pos <= answer_end_pos[data_i]
                ):
                    hidden_states = outputs.hidden_states

                    if cur_pos == answer_begin_pos[data_i]:
                        # initialize the output tensor
                        for temp_idx, layer_idx in enumerate(self.layer_list):
                            if self.output_token_average_hidden_states:
                                output_average_tensor[data_i, temp_idx] = (
                                    hidden_states[layer_idx][data_i, -1, :]
                                )
                            if self.len_of_token_hidden_states_output > 0:
                                output_last_token_tensor[
                                    data_i, temp_idx
                                ] = hidden_states[layer_idx][
                                    data_i,
                                    cur_pos
                                    - self.len_of_token_hidden_states_output : cur_pos,
                                    :,
                                ]
                            if self.get_query_entropies:
                                probs = F.softmax(
                                    outputs.logits[:, -1, :], dim=-1
                                )
                                entropy = -torch.sum(
                                    probs * torch.log(probs + 1e-10), dim=-1
                                )
                                entropy_list[data_i] = [
                                    entropy[data_i],
                                ]
                                entropy_output_tensor[
                                    data_i, self.max_entropy_idx
                                ] = entropy[
                                    data_i
                                ]  # max entropy
                                entropy_output_tensor[
                                    data_i, self.min_entropy_idx
                                ] = entropy[
                                    data_i
                                ]  # min entropy
                                entropy_output_tensor[
                                    data_i, self.mean_entropy_idx
                                ] = entropy[
                                    data_i
                                ]  # mean entropy
                                entropy_output_tensor[
                                    data_i, self.std_entropy_idx
                                ] = 0  # std entropy
                            if self.get_query_probs:
                                probs = F.softmax(
                                    outputs.logits[:, -1, :], dim=-1
                                )
                                prob_list[data_i] = probs[
                                    data_i, next_toks[data_i]
                                ]

                    else:
                        # update the output tensor
                        for temp_idx, layer_idx in enumerate(self.layer_list):
                            n = cur_pos - answer_begin_pos[data_i] + 1
                            if self.output_token_average_hidden_states:
                                output_average_tensor[data_i, temp_idx] = (
                                    output_average_tensor[data_i, temp_idx]
                                    * (n - 1)
                                    + hidden_states[layer_idx][data_i, -1, :]
                                ) / n
                            if self.len_of_token_hidden_states_output > 0:
                                output_last_token_tensor[
                                    data_i,
                                    temp_idx,
                                    0 : self.len_of_token_hidden_states_output
                                    - 1,
                                    :,
                                ] = output_last_token_tensor[
                                    data_i,
                                    temp_idx,
                                    1 : self.len_of_token_hidden_states_output,
                                    :,
                                ].clone()
                                output_last_token_tensor[
                                    data_i,
                                    temp_idx,
                                    self.len_of_token_hidden_states_output - 1,
                                    :,
                                ] = hidden_states[layer_idx][data_i, -1, :]
                            if self.get_query_entropies:
                                probs = F.softmax(
                                    outputs.logits[:, -1, :], dim=-1
                                )
                                entropy = -torch.sum(
                                    probs * torch.log(probs + 1e-10), dim=-1
                                )
                                entropy_list[data_i].append(entropy[data_i])
                                entropy_output_tensor[
                                    data_i, self.max_entropy_idx
                                ] = torch.max(
                                    entropy_output_tensor[
                                        data_i, self.max_entropy_idx
                                    ],
                                    entropy[data_i],
                                )
                                entropy_output_tensor[
                                    data_i, self.min_entropy_idx
                                ] = torch.min(
                                    entropy_output_tensor[
                                        data_i, self.min_entropy_idx
                                    ],
                                    entropy[data_i],
                                )
                                old_mean = entropy_output_tensor[
                                    data_i, self.mean_entropy_idx
                                ]
                                entropy_output_tensor[
                                    data_i, self.mean_entropy_idx
                                ] = (
                                    entropy_output_tensor[
                                        data_i, self.mean_entropy_idx
                                    ]
                                    * (n - 1)
                                    + entropy[data_i]
                                ) / n
                                entropy_output_tensor[
                                    data_i, self.std_entropy_idx
                                ] = torch.std(
                                    torch.tensor(entropy_list[data_i])
                                )
                            if self.get_query_probs:
                                probs = F.softmax(
                                    outputs.logits[:, -1, :], dim=-1
                                )
                                prob = probs[data_i, next_toks[data_i]]
                                prob_list[data_i] = torch.cat(
                                    [
                                        prob_list[data_i].reshape(1, -1),
                                        prob.reshape(1, -1),
                                    ],
                                    dim=1,
                                )

                                prob_output_tensor[data_i, 0] = torch.max(
                                    -prob_list[data_i]
                                )
                                prob_output_tensor[data_i, 1] = torch.min(
                                    -prob_list[data_i]
                                )
                                prob_output_tensor[data_i, 2] = torch.mean(
                                    -prob_list[data_i]
                                )
                                prob_output_tensor[data_i, 3] = torch.std(
                                    -prob_list[data_i]
                                )
                                prob_output_tensor[data_i, 4] = torch.mean(
                                    -torch.log(prob_list[data_i] + 1e-10)
                                )
                                prob_output_tensor[data_i, 5] = torch.std(
                                    -torch.log(prob_list[data_i] + 1e-10)
                                )

            tokens[:, cur_pos] = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_toks
            )
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[answer_begin_pos[i] : len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: answer_end_pos[i] - answer_begin_pos[i]]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))

        if self.output_token_average_hidden_states:

            return (
                decoded,
                output_average_tensor,
                output_last_token_tensor,
                entropy_output_tensor,
                prob_output_tensor,
            )
        else:
            return decoded

    def sample_next(
        self,
        logits: torch.FloatTensor,  # (bsz, vocab_size): logits for last token
        temperature: float,  # temperature for sampling
        top_p: float,  # top p for sampling
    ) -> torch.LongTensor:
        """Vanilla sampling with temperature and top p."""

        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            next_token = torch.multinomial(
                probs_sort, num_samples=1
            )  # one hot of next token, ordered by original probs
            next_token = torch.gather(
                probs_idx, -1, next_token
            )  # one hot of next token, ordered by vocab
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        return next_token
