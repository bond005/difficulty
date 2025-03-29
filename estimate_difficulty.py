from argparse import ArgumentParser
import codecs
import copy
import json
import os
import random
import sys
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

try:
    from difficulty.io_utils import load_samples
    from difficulty.crossentropy import calculate_crossentropy
except:
    sys.path.append(os.path.dirname(__file__))
    from difficulty.io_utils import load_samples
    from difficulty.crossentropy import calculate_crossentropy


DEFAULT_RANDOM_SEED: int = 42


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available!')
    random.seed(DEFAULT_RANDOM_SEED)
    torch.manual_seed(DEFAULT_RANDOM_SEED)
    np.random.seed(DEFAULT_RANDOM_SEED)
    torch.cuda.manual_seed(DEFAULT_RANDOM_SEED)

    parser = ArgumentParser()
    parser.add_argument('-i', '--input', dest='input_name', type=str, required=True,
                        help='The input dataset name.')
    parser.add_argument('-o', '--output', dest='output_name', type=str, required=True,
                        help='The output dataset name.')
    parser.add_argument('-m', '--model', dest='model_name', type=str, required=True,
                        help='The causal language model name.')
    parser.add_argument('--max_seq_len', dest='max_seq_len', type=int, required=False,
                        default=None, help='The maximal sequence length.')
    args = parser.parse_args()

    input_fname = os.path.normpath(args.input_name)
    if not os.path.isfile(input_fname):
        raise IOError(f'The file "{input_fname}" does not exist!')

    output_fname = os.path.normpath(args.output_name)
    if not os.path.isfile(output_fname):
        basedir = os.path.dirname(output_fname)
        if len(basedir) > 0:
            if not os.path.isdir(basedir):
                raise IOError(f'The directory "{basedir}" does not exist!')

    if args.max_seq_len is None:
        estimated_samples = load_samples(input_fname)
    else:
        estimated_samples = load_samples(input_fname, max_line_size=3 * args.max_seq_len)
    if len(estimated_samples) == 0:
        raise IOError(f'The file "{input_fname}" is empty!')
    print(f'There are {len(estimated_samples)} estimated samples.')

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype='auto',
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    with codecs.open(output_fname, mode='w', encoding='utf-8', buffering=0) as fp:
        for idx, val in enumerate(tqdm(estimated_samples)):
            if 'query' in val:
                res = calculate_crossentropy(
                    tokenizer, model, user_prompt=val['query'],
                    system_prompt=val.get('system', None),
                    history=val.get('history', None),
                    reference=val['response'],
                    max_seq_len=args.max_seq_len
                )
            else:
                res = calculate_crossentropy(
                    tokenizer, model, user_prompt=val['response'],
                    max_seq_len=args.max_seq_len
                )
            if res is None:
                print(f'The sample {idx} is too large.')
            else:
                seq_len, loss_val = res
                new_sample: Dict[str, Union[str, int, float, List[Tuple[str, str]]]] = copy.deepcopy(val)
                new_sample['sequence_length'] = seq_len
                new_sample['loss_function'] = loss_val
                fp.write(json.dumps(new_sample, ensure_ascii=False) + '\n')
                del res, new_sample
    print(f'Difficulty of all samples is successfully estimated.')


if __name__ == '__main__':
    main()
