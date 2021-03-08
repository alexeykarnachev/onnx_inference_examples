import datetime

import numpy as np
from transformers import GPT2TokenizerFast
import tritonhttpclient

_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
_MODEL_NAME = 'distilgpt2_past'
_MODEL_VERSION = '1'
_URL = 'triton_server:8000'

_N_LAYERS = 6
input_names = ['input_ids', 'position_ids', 'attention_mask'] + [f'past_{i}' for i in range(_N_LAYERS)]
output_names = ['logits'] + [f'present_{i}' for i in range(_N_LAYERS)]

_N_REPEATS = 100


def _prepare_input_ids(input_ids):
    input_ids = np.array(input_ids, dtype=np.int64).reshape(1, len(input_ids))
    input_ids_prepared = tritonhttpclient.InferInput('input_ids', input_ids.shape, 'INT64')
    input_ids_prepared.set_data_from_numpy(input_ids, binary_data=False)

    return input_ids_prepared


def _prepare_position_ids(position_ids):
    position_ids = np.array(position_ids, dtype=np.int64).reshape(1, len(position_ids))
    position_ids_prepared = tritonhttpclient.InferInput('position_ids', position_ids.shape, 'INT64')
    position_ids_prepared.set_data_from_numpy(position_ids, binary_data=False)

    return position_ids_prepared


def _prepare_attention_mask():
    attention_mask = np.array([[1]], dtype=np.float32)
    attention_mask_prepared = tritonhttpclient.InferInput('attention_mask', attention_mask.shape, 'FP32')
    attention_mask_prepared.set_data_from_numpy(attention_mask, binary_data=False)

    return attention_mask_prepared


def _prepare_past(ind):
    past = np.random.randn(*[2, 1, 12, 2, 64]).astype(np.float32)
    past_prepared = tritonhttpclient.InferInput(f'past_{ind}', past.shape, 'FP32')
    past_prepared.set_data_from_numpy(past, binary_data=False)

    return past_prepared


def _prepare_inputs(context):
    # input_ids = _tokenizer.encode(context, add_special_tokens=False)
    input_ids = [228]
    input_ids_prepared = _prepare_input_ids(input_ids)

    position_ids = list(range(len(input_ids)))
    position_ids_prepared = _prepare_position_ids(position_ids)

    attention_mask_prepared = _prepare_attention_mask()

    pasts_prepared = [_prepare_past(ind) for ind in range(_N_LAYERS)]

    inputs_prepared = [input_ids_prepared, position_ids_prepared, attention_mask_prepared] + pasts_prepared

    return inputs_prepared


def _prepare_outputs():
    outputs = [tritonhttpclient.InferRequestedOutput(output_name, binary_data=False) for output_name in output_names]

    return outputs


def run_inference(context, model_name=_MODEL_NAME, url=_URL, model_version=_MODEL_VERSION):
    triton_client = tritonhttpclient.InferenceServerClient(url=url, verbose=False)
    # model_metadata = triton_client.get_model_metadata(model_name=model_name, model_version=model_version)
    # model_config = triton_client.get_model_config(model_name=model_name, model_version=model_version)

    inputs = _prepare_inputs(context)
    outputs = _prepare_outputs()

    start = datetime.datetime.now()
    for i in range(_N_REPEATS):
        response = triton_client.infer(model_name, model_version=model_version, inputs=inputs, outputs=outputs)
        # present_0 = response.as_numpy('present_0')
        # present_0 = np.asarray(present_0, dtype=np.float32)
        print(i)
    end = datetime.datetime.now()
    delta = end - start
    time_per_call = delta / _N_REPEATS
    print(time_per_call)


if __name__ == '__main__':
    run_inference('Wotafak mazafak')
