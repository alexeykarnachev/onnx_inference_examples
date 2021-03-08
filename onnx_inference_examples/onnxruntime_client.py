import collections.abc
import datetime

import numpy as np
import onnxruntime
from onnxruntime.transformers.gpt2_helper import Gpt2Helper, Gpt2Inputs
import torch
from transformers import GPT2Config, GPT2TokenizerFast

from logits_modifiers import IgnoredTokensModifier, TemperatureModifier, TopKNucleusModifier


class GPT2Onnx:
    def __init__(self, onnx_file_path, model_name_or_path, dialog_turn_token, max_n_tokens, max_n_context_tokens,
                 is_float16, cuda_id):
        assert max_n_tokens > max_n_context_tokens, 'max_n_tokens must be > max_n_context_tokens'

        self._session = onnxruntime.InferenceSession(onnx_file_path)
        self._input_names = [inp.name for inp in self._session.get_inputs()]
        self._output_names = [out.name for out in self._session.get_outputs()]
        self._dialog_turn_token = dialog_turn_token
        self._max_n_tokens = max_n_tokens
        self._max_n_context_tokens = max_n_context_tokens
        self._tokenizer = GPT2TokenizerFast.from_pretrained(model_name_or_path)
        self._gpt2_config = GPT2Config.from_pretrained(model_name_or_path)
        self._n_layers = self._gpt2_config.num_hidden_layers
        self._cuda_id = cuda_id
        self._device = f'cuda:{cuda_id}'
        self._is_float16 = is_float16
        self._torch_float_type = torch.float16 if self._is_float16 else torch.float32
        self._io_binding = self._session.io_binding()

    def generate_candidates(self, context, n_candidates):
        prefix = self._prepare_prefix(context)
        encoded_prefix = self._tokenizer.encode(prefix, add_special_tokens=False)
        encoded_prefix = encoded_prefix[-self._max_n_context_tokens:]
        prefix_seq_len = len(encoded_prefix)
        encoded_prefixes = [encoded_prefix[:] for _ in range(n_candidates)]

        max_output_shapes = Gpt2Helper.get_output_shapes(
            batch_size=n_candidates,
            past_sequence_length=self._max_n_tokens,
            sequence_length=self._max_n_tokens,
            config=self._gpt2_config)

        input_ids = torch.tensor(encoded_prefixes, dtype=torch.long, device=self._device)
        position_ids = torch.arange(prefix_seq_len, dtype=torch.long, device=self._device).repeat((n_candidates, 1))
        past = self._get_empty_past(n_candidates)
        attention_mask = torch.ones((n_candidates, prefix_seq_len), dtype=self._torch_float_type, device=self._device)
        gpt2_inputs = Gpt2Inputs(input_ids, position_ids, attention_mask, past)

        output_buffers = Gpt2Helper.get_output_buffers(max_output_shapes, self._device, is_float16=self._is_float16)
        output_shapes = Gpt2Helper.get_output_shapes(
            batch_size=n_candidates, past_sequence_length=0, sequence_length=prefix_seq_len, config=self._gpt2_config)
        ort_outputs = Gpt2Helper.onnxruntime_inference_with_binded_io(
            ort_session=self._session,
            inputs=gpt2_inputs,
            output_buffers=output_buffers,
            output_shapes=output_shapes,
            return_numpy=False)

        # ================================================================================
        # Continue inference after the first step (using past):
        generated_input_ids = []

        for i_step in range(24):
            next_token_logits = ort_outputs[0][:, -1, :].double()

            _modify_next_token_logits(next_token_logits, ignored_input_ids=None, temperature=0.7, top_k=50, top_p=1.0)

            next_input_ids = _sample_next_input_ids(next_token_logits)
            generated_input_ids.append(next_input_ids)

            past = ort_outputs[1:]

            input_ids = next_input_ids[:, None]
            past_sequence_length = past[0].size(3)
            new_seq_len = 1 + past_sequence_length

            position_ids = torch.tensor(
                [new_seq_len - 1], dtype=torch.long, device=self._device).repeat((n_candidates, 1))
            attention_mask = torch.ones((n_candidates, new_seq_len), dtype=self._torch_float_type, device=self._device)

            gpt2_inputs = Gpt2Inputs(input_ids, position_ids, attention_mask, past)
            output_shapes = Gpt2Helper.get_output_shapes(
                batch_size=n_candidates,
                past_sequence_length=past_sequence_length,
                sequence_length=1,
                config=self._gpt2_config)

            ort_outputs = Gpt2Helper.onnxruntime_inference_with_binded_io(
                ort_session=self._session,
                inputs=gpt2_inputs,
                output_buffers=output_buffers,
                output_shapes=output_shapes,
                return_numpy=False)

        generated_input_ids = torch.stack(generated_input_ids)

        generated_input_ids = generated_input_ids.cpu().numpy().T.tolist()
        decoded_candidates = self._tokenizer.batch_decode(generated_input_ids)

        return decoded_candidates

    def _prepare_prefix(self, context):
        assert isinstance(context, collections.abc.Sequence) and not isinstance(context, str)
        prefix = self._dialog_turn_token.join(context) + self._dialog_turn_token

        return prefix

    def _get_empty_past(self, n_candidates):
        past = [
            torch.randn(
                *[
                    2, n_candidates, self._gpt2_config.num_attention_heads, 0,
                    int(self._gpt2_config.n_embd / self._gpt2_config.n_head)
                ],
                device=self._device,
                dtype=self._torch_float_type) for _ in range(self._n_layers)
        ]

        return past


def _sample_next_input_ids(next_token_logits):
    probabilities = torch.nn.functional.softmax(next_token_logits, dim=-1)
    next_tokens = torch.multinomial(probabilities, num_samples=1)
    return next_tokens.squeeze(1)


def _modify_next_token_logits(next_token_logits, ignored_input_ids, temperature, top_k, top_p):
    modifiers = [
        IgnoredTokensModifier(ignored_input_ids=ignored_input_ids),
        TemperatureModifier(temperature=temperature),
        TopKNucleusModifier(top_k=top_k, top_p=top_p)
    ]

    _ = [modifier(next_token_logits) for modifier in modifiers]


if __name__ == '__main__':
    model = GPT2Onnx('/workspace/onnx_inference_examples/gpt2-large/gpt2-large_past_fp16/gpt2-large_past_fp16.onnx', 'gpt2-large',
            '->', 50, 25, True, 0)
    context = ["What is your name?", "My name is Alex", "How old are you?"] * 100
    n_repeats = 1000
    n_candidates = 8
    start_time = datetime.datetime.now()
    for i in range(n_repeats):
        candidates = model.generate_candidates(context, n_candidates)
        print(i)
        print(candidates[0])


    end_time = datetime.datetime.now()
    delta = (end_time - start_time) / n_repeats
    print(delta)
