import collections
import datetime

import numpy as np
import onnxruntime
from onnxruntime.transformers.gpt2_helper import Gpt2Helper, Gpt2Inputs
import torch
from transformers import GPT2Config, GPT2TokenizerFast, top_k_top_p_filtering


class GPT2Onnx:
    def __init__(self, onnx_file_path, model_name_or_path, dialog_turn_token, dialog_pad_token, dialog_eos_token,
                 max_full_len, max_context_len, max_batch_size, is_float16, cuda_id):
        assert max_full_len > max_context_len

        self._onnx_file_path = onnx_file_path
        self._model_name_or_path = model_name_or_path
        self._dialog_turn_token = dialog_turn_token
        self._dialog_pad_token = dialog_pad_token
        self._dialog_eos_token = dialog_eos_token
        self._max_full_len = max_full_len
        self._max_context_len = max_context_len
        self._max_batch_size = max_batch_size
        self._is_float16 = is_float16
        self._cuda_id = cuda_id

        # Huggingface:
        self._gpt2_tokenizer = GPT2TokenizerFast.from_pretrained(model_name_or_path)
        self._gpt2_config = GPT2Config.from_pretrained(model_name_or_path)

        # Device and fp:
        self._cuda_device = torch.device('cuda', self._cuda_id)
        self._cuda_float_type = torch.float16 if self._is_float16 else torch.float32

        # Special token ids:
        self._dialog_eos_token_id = self._gpt2_tokenizer.convert_tokens_to_ids(dialog_eos_token)
        self._dialog_turn_token_id = self._gpt2_tokenizer.convert_tokens_to_ids(dialog_turn_token)
        self._dialog_pad_token_id = self._gpt2_tokenizer.convert_tokens_to_ids(dialog_pad_token)
        self._input_ids_not_to_penalize = torch.tensor(
            [self._dialog_eos_token_id, self._dialog_turn_token_id], dtype=torch.long, device=self._cuda_device)

        # ONNX session related stuff:
        self._onnx_session = onnxruntime.InferenceSession(onnx_file_path)
        self._onnx_input_names = [inp.name for inp in self._onnx_session.get_inputs()]
        self._onnx_output_names = [out.name for out in self._onnx_session.get_outputs()]
        self._onnx_io_binding = self._onnx_session.io_binding()
        self._onnx_max_output_shapes = Gpt2Helper.get_output_shapes(
            batch_size=self._max_batch_size,
            past_sequence_length=self._max_full_len,
            sequence_length=self._max_full_len,
            config=self._gpt2_config)
        self._onnx_output_buffers = Gpt2Helper.get_output_buffers(
            output_shapes=self._onnx_max_output_shapes, device=self._cuda_device, is_float16=self._is_float16)

    def generate_responses(self, contexts, temperature, top_k, top_p, repetition_penalty, ignored_input_ids):
        assert len(contexts) <= self._max_batch_size
 
        # Get initial gpt2 inputs:
        context_input_ids = self._get_input_ids(contexts)
        batch_size, context_len = context_input_ids.size()
        position_ids = torch.arange(context_len, dtype=torch.long, device=self._cuda_device).repeat((batch_size, 1))
        attention_mask = torch.ones((batch_size, context_len), dtype=self._cuda_float_type, device=self._cuda_device)
        past = self._get_empty_past(batch_size)
        gpt2_inputs = Gpt2Inputs(
            input_ids=context_input_ids, position_ids=position_ids, attention_mask=attention_mask, past=past)
 
        # Get initial onnx outputs shapes:
        onnx_output_shapes = Gpt2Helper.get_output_shapes(
            batch_size=batch_size,
            past_sequence_length=0,  # No past values at first generation step.
            sequence_length=context_len,
            config=self._gpt2_config)

        # Get initial onnx outputs:
        onnx_outputs = Gpt2Helper.onnxruntime_inference_with_binded_io(
            ort_session=self._onnx_session,
            inputs=gpt2_inputs,
            output_buffers=self._onnx_output_buffers,
            output_shapes=onnx_output_shapes,
            return_numpy=False)

        # Continue inference after the first step (Now, usign past cached values):
        max_generated_len = self._max_full_len - context_len 
        generated_input_ids = torch.full((batch_size, max_generated_len), self._dialog_eos_token_id, dtype=torch.long, device=self._cuda_device)
        sample_is_finished_mask = torch.zeros(batch_size, dtype=torch.bool, device=self._cuda_device)
        generated_len = torch.zeros_like(sample_is_finished_mask, dtype=torch.long)
 
        n_steps_done = 0
        while n_steps_done < max_generated_len:
            # Extract vectors from onnx outputs:
            past = onnx_outputs[1:]
            next_input_logits = onnx_outputs[0][:, -1, :].float()

            # Modify logits:
#            _penalize_logits(
#                next_input_logits,
#                generated_input_ids,
#                input_ids_not_to_penalize=self._input_ids_not_to_penalize,
#                penalty=repetition_penalty)  # Penalize prev.
#            _penalize_logits(
#                next_input_logits,
#                ignored_input_ids,
#                input_ids_not_to_penalize=self._input_ids_not_to_penalize,
#                penalty=float('inf'))  # Penalize ignored
            _change_logits_temperature(next_input_logits, temperature)
            _change_logits_nucleus(next_input_logits, top_k=top_k, top_p=top_p)

            # Sample new tokes:
            next_input_ids = _sample_input_ids(next_input_logits)
            generated_input_ids[:, n_steps_done] = next_input_ids
            n_steps_done += 1
            

            # Actualize is-finished mask and generated samples' input_ids lengths:
            sample_is_finished_mask |= next_input_ids == self._dialog_eos_token_id
            generated_len += ~sample_is_finished_mask

            # Check if the generation is finished. It's finished if:
            # 1. We've reached the max number of input_ids to be generated, or
            # 2. If each sample in batch have reached the eos token.
            if n_steps_done == max_generated_len:
                break

            if sample_is_finished_mask.all():
                break

            

            # Prepare new inference inputs:
            past_seq_len = past[0].size(3)
            next_position_ids = gpt2_inputs.position_ids[:, -1:] + 1  # Increment position ids
            next_attention_mask = torch.cat(
                [
                    gpt2_inputs.attention_mask,
                    torch.ones((batch_size, 1), dtype=self._cuda_float_type, device=self._cuda_device),
                ],
                dim=1)

            gpt2_inputs = Gpt2Inputs(
                input_ids=next_input_ids[:, None],
                position_ids=next_position_ids,
                attention_mask=next_attention_mask,
                past=past)
            onnx_output_shapes = Gpt2Helper.get_output_shapes(
                batch_size=batch_size,
                past_sequence_length=past_seq_len,
                sequence_length=1,  # On new inference step we put only one new token.
                config=self._gpt2_config)

            # Perform new step inference:
            onnx_outputs = Gpt2Helper.onnxruntime_inference_with_binded_io(
                ort_session=self._onnx_session,
                inputs=gpt2_inputs,
                output_buffers=self._onnx_output_buffers,
                output_shapes=onnx_output_shapes,
                return_numpy=False)
        
        # Decode responses:
        generated_input_ids = generated_input_ids.cpu().numpy()
        generated_len = generated_len.cpu().numpy()
        decoded_responses = []
        
        for i_response in range(len(generated_input_ids)):
            response_len = generated_len[i_response]
            decoded_response = self._gpt2_tokenizer.decode(generated_input_ids[i_response, :response_len])
            decoded_responses.append(decoded_response)
        
        return decoded_responses

    def _get_input_ids(self, contexts):
        context_strings = []
        for context in contexts:
            assert isinstance(context, collections.abc.Sequence) and not isinstance(context, str)
            context_string = self._dialog_turn_token.join(context) + self._dialog_turn_token
            context_strings.append(context_string)

        encoded_contexts = self._gpt2_tokenizer.batch_encode_plus(
            context_strings, add_special_tokens=False)['input_ids']
        max_n_input_ids = max(len(encoded_context) for encoded_context in encoded_contexts)
        max_n_input_ids = min(max_n_input_ids, self._max_context_len)
        input_ids = np.zeros((len(contexts), max_n_input_ids))
        input_ids.fill(self._dialog_pad_token_id)

        for i_sample, encoded_context in enumerate(encoded_contexts):
            encoded_context = encoded_context[-max_n_input_ids:]
            input_ids[i_sample, -len(encoded_context):] = encoded_context

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self._cuda_device)

        return input_ids

    def _get_empty_past(self, n_candidates):
        seq_len = 0  # Empy past means empty seq len. Used for initial forward pass.

        past = [
            torch.randn(
                *[
                    2, n_candidates, self._gpt2_config.num_attention_heads, seq_len,
                    int(self._gpt2_config.n_embd / self._gpt2_config.n_head)
                ],
                device=self._cuda_device,
                dtype=self._cuda_float_type) for _ in range(self._gpt2_config.num_hidden_layers)
        ]

        return past


def _penalize_logits(logits, input_ids_to_penalize, input_ids_not_to_penalize, penalty):
    """Penalization procedure in huggingface or in CTRL paper is wrong.
    This is the correct one.

    Note, that the function modifies logits inplace (and also returns).

    :param logits: Tensor, (batch_size, vocab_size).
    :param input_ids_to_penalize: Tensor, contains token ids to be penalized. It could be 1d, 2d or any-d
        tensor. The function will take unique values from it.
    :param input_ids_not_to_penalize: Tensor, contains token ids which will NOT be penalized, even if they
        present in `input_ids_to_penalize` tensor. It's convinient to use such the tensor to preserve
        special input_ids (eos or dialog turn input_ids) from penalization.
    :param penalty: Float, penalization penalty. The final probability of each penalizable token will be
        decreased by `penalty` times.
    :return: Tensor, modified logits.
    """
    assert penalty >= 1.0, 'Penalization penalty must be >= 1.0'
    if penalty == 1.0 or input_ids_to_penalize is None or not (len(input_ids_to_penalize)):
        return

    # Select input_ids to penalize:
    not_to_penalize_idx = torch.unique(input_ids_not_to_penalize)
    penalize_idx = torch.unique(input_ids_to_penalize)
    penalize_idx_mask = (~penalize_idx.unsqueeze(1).eq(not_to_penalize_idx)).all(-1)
    penalize_idx = penalize_idx[penalize_idx_mask]

    if len(penalize_idx) == 0:
        return logits
    
    if penalty == float('inf'):
        logits[penalize_idx] = -9999
    else:
        logits -= logits.max()
        full_exp = torch.exp(logits)

        e = full_exp[penalize_idx]
        sum_e = torch.sum(e)
        s = torch.sum(full_exp) - sum_e

        n = torch.log((e * s) / (penalty * s + penalty * sum_e - sum_e))
        logits[penalize_idx] = n

    return logits


def _change_logits_temperature(logits, temperature):
    """
    :param logits: Tensor, (batch_size, vocab_size).
    :param temperature: Float, temperature value (the value to devide logits on).
    :return: Tensor, modified logits.
    """
    logits.mul_(1 / temperature)

    return logits


def _change_logits_nucleus(logits, top_k, top_p):
    """Delegates call to the transformers `top_k_top_p_filtering` func."""

    return top_k_top_p_filtering(logits=logits, top_k=top_k, top_p=top_p)


def _sample_input_ids(logits):
    """Samples input ids from logits. Logits are assumed to be already modified (temperature, penalization).

    :param logits: Tensor, (batch_size, vocab_size).
    :return: Tensor, with sampled token ids (batch_size,).
    """
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    next_input_ids = torch.multinomial(probabilities, num_samples=1)

    return next_input_ids.squeeze(1)


if __name__ == '__main__':
    model = GPT2Onnx(
        onnx_file_path='/workspace/onnx_inference_examples/gpt2-large/gpt2-large_past_fp16/gpt2-large_past_fp16.onnx',
        model_name_or_path='gpt2-large',
        dialog_turn_token='->',
        dialog_pad_token='<|endoftext|>',
        dialog_eos_token='Ъ',
        max_full_len=50,
        max_context_len=25,
        max_batch_size=32,
        is_float16=True,
        cuda_id=0)
    contexts = [['Hello', 'Hi nigga, how are you'] * 100 for _ in range(8)]

    n_repeats = 1000
    start_time = datetime.datetime.now()
    for i in range(n_repeats):
        candidates = model.generate_responses(
        contexts=contexts, temperature=0.8, top_k=50, top_p=1.0, repetition_penalty=3, ignored_input_ids=None)
        print(i)
        print(candidates[0:2])


    end_time = datetime.datetime.now()
    delta = (end_time - start_time) / n_repeats
    print(delta)
