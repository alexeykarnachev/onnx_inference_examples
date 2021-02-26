import datetime

import numpy as np
import onnxruntime

_BS = 8
_PAST_SEQ_LEN = 50
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


if __name__ == '__main__':
    main()
