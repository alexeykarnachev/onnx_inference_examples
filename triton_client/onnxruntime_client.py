import collections.abc
import datetime

import numpy as np
import onnxruntime
from transformers import GPT2TokenizerFast

_BS = 8
_PAST_SEQ_LEN = 0
_N_REPEATS = 500
_USE_FP16 = True
_FLOAT_TYPE = np.float16 if _USE_FP16 else np.float32


def main():
    session = onnxruntime.InferenceSession("/workspace/distilgpt2.onnx")
    inputs = session.get_inputs()
    outputs = session.get_outputs()

    input_names = [inp.name for inp in inputs]
    output_names = [output.name for output in outputs]
    input_ids = np.random.randint(low=0, high=100, size=(_BS, 1)).astype(np.int64)
    position_ids = np.zeros_like(input_ids)
    attention_mask = np.ones(shape=(_BS, 1), dtype=_FLOAT_TYPE)
    inputs = {'input_ids': input_ids, 'position_ids': position_ids, 'attention_mask': attention_mask}
    pasts = {
        name: np.random.randn(*[2, _BS, 12, _PAST_SEQ_LEN, 64]).astype(_FLOAT_TYPE)
        for name in input_names
        if 'past' in name
    }
    inputs.update(pasts)

    start = datetime.datetime.now()
    for i in range(_N_REPEATS):
        _ = session.run(output_names, inputs)
        print(i)
    end = datetime.datetime.now()
    delta = end - start
    time_per_call = delta / _N_REPEATS
    time_per_call_ms = time_per_call.microseconds / 1000
    print(time_per_call_ms)


class GPT2Onnx:
    def __init__(self, onnx_file_path, tokenizer_name_or_path, dialog_turn_token, max_n_tokens, max_n_context_tokens,
                 use_fp16, cuda_id):
        assert max_n_tokens > max_n_context_tokens, 'max_n_tokens must be > max_n_context_tokens'

        self._session = onnxruntime.InferenceSession(onnx_file_path)
        self._input_names = [inp.name for inp in self._session.get_inputs()]
        self._output_names = [out.name for out in self._session.get_outputs()]
        self._dialog_turn_token = dialog_turn_token
        self._max_n_tokens = max_n_tokens
        self._max_n_context_tokens = max_n_context_tokens
        self._tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name_or_path)
        self._float_type = np.float16 if use_fp16 else np.float32
        self._cuda_id = cuda_id

        self._io_binding = self._session.io_binding()

    def generate_candidates(self, context, n_candidates):
        assert isinstance(context, collections.abc.Sequence) and not isinstance(context, str)
        prefix = self._dialog_turn_token.join(context)
        encoded_prefix = self._tokenizer.encode(prefix, add_special_tokens=False)
        encoded_prefix = encoded_prefix[-self._max_n_context_tokens:]
        encoded_prefixes = [encoded_prefix[:] for _ in range(n_candidates)]

        input_ids = np.array(encoded_prefixes, dtype=np.int64)
        position_ids = np.tile(np.arange(len(encoded_prefix)), (n_candidates, 1))
        past = self._get_empty_past(n_candidates)
        attention_mask = np.ones((n_candidates, input_ids.shape[1]), dtype=self._float_type)

        inputs = {'input_ids': input_ids, 'position_ids': position_ids, 'attention_mask': attention_mask}
        inputs.update(past)

        for input_name, input_numpy_arr in inputs.items():
            input_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(input_numpy_arr, 'cuda', 0)
            self._io_binding.bind_ortvalue_input(input_name, input_ortvalue)

        for output_name in self._output_names:
            self._io_binding.bind_output(output_name, 'cuda', 0)

        self._session.run_with_iobinding(self._io_binding)
        output_ortvalue = self._io_binding.get_outputs()[0]

        print(output_ortvalue.device_name())

    def _get_empty_past(self, n_candidates):
        past = {
            name: np.random.randn(*[2, n_candidates, 12, 0, 64]).astype(self._float_type)
            for name in self._input_names
            if 'past' in name
        }

        return past


if __name__ == '__main__':
    model = GPT2Onnx("/workspace/distilgpt2.onnx", 'gpt2', '->', 100, 50, True, 0)
    context = ["What is your name?", "My name is Alex", "How old are you?"]
    n_candidates = 8
    model.generate_candidates(context, n_candidates)
