import codecs
import json
from typing import Dict, List, Optional, Tuple, Union
import warnings


def load_samples(fname: str,
                 max_line_size: Optional[int] = None) -> List[Dict[str, Union[str, List[Tuple[str, str]]]]]:
    line_idx = 1
    samples = []
    with codecs.open(fname, mode='r', encoding='utf-8', errors='ignore') as fp:
        curline = fp.readline()
        while len(curline) > 0:
            prepline = curline.strip()
            if len(prepline) > 0:
                if (max_line_size is not None) and (len(prepline) > max_line_size):
                    warn_msg = (f'File "{fname}": line {line_idx} is too long! Expected {max_line_size} or less, '
                                f'got {len(prepline)}.')
                    warnings.warn(warn_msg)
                else:
                    err_msg = f'File "{fname}": line {line_idx} is wrong!'
                    try:
                        loaded_sample = json.loads(prepline)
                        warn_msg = ''
                    except Exception as err:
                        loaded_sample = None
                        warn_msg = err_msg + ' JSON cannot be parsed. ' + str(err)
                    if len(warn_msg) == 0:
                        warnings.warn(warn_msg)
                    elif not isinstance(loaded_sample, dict):
                        warn_msg = err_msg + f' Expected {type({"a": 1})}, got {type(loaded_sample)}.'
                        warnings.warn(warn_msg)
                        del loaded_sample
                    else:
                        if 'response' not in loaded_sample:
                            err_msg += f' The keys {list(loaded_sample.keys())} are wrong!'
                            raise IOError(err_msg)
                        if not isinstance(loaded_sample['response'], str):
                            err_msg += (f' The response is wrong! Expected {type("a")}, '
                                        f'got {type(loaded_sample["response"])}.')
                            raise IOError(err_msg)
                        if len(loaded_sample['response'].strip()) == 0:
                            err_msg += ' The response is empty!'
                            raise IOError(err_msg)
                        new_sample: Dict[str, Union[str, List[Tuple[str, str]]]] = {
                            'response': loaded_sample['response']
                        }
                        if 'system' in loaded_sample:
                            if not isinstance(loaded_sample['system'], str):
                                err_msg += (f' The system is wrong! Expected {type("a")}, '
                                            f'got {type(loaded_sample["system"])}.')
                                raise IOError(err_msg)
                            new_sample['system'] = loaded_sample['system']
                        if 'query' in loaded_sample:
                            if not isinstance(loaded_sample['query'], str):
                                err_msg += (f' The query is wrong! Expected {type("a")}, '
                                            f'got {type(loaded_sample["query"])}.')
                                raise IOError(err_msg)
                            if len(loaded_sample['query'].strip()) == 0:
                                err_msg += ' The query is empty!'
                                raise IOError(err_msg)
                            new_sample['query'] = loaded_sample['query']
                        if ('system' in loaded_sample) and ('query' not in loaded_sample):
                            err_msg += ' The query is not found!'
                            raise IOError(err_msg)
                        if 'history' in loaded_sample:
                            if 'query' not in loaded_sample:
                                err_msg += ' The query is not found!'
                                raise IOError(err_msg)
                            if not isinstance(loaded_sample['history'], list):
                                err_msg += (f' The history is wrong! Expected {type([1, 2])}, '
                                            f'got {type(loaded_sample["history"])}.')
                                raise IOError(err_msg)
                            history: List[Tuple[str, str]] = []
                            for idx, val in enumerate(loaded_sample['history']):
                                if not isinstance(val, list):
                                    err_msg += (f' The history[{idx}] is wrong! Expected {type([1, 2])}, '
                                                f'got {type(val)}.')
                                    raise IOError(err_msg)
                                if len(val) != 2:
                                    err_msg += f' The history[{idx}] is wrong!'
                                    raise IOError(err_msg)
                                if (not isinstance(val[0], str)) or (not isinstance(val[1], str)):
                                    err_msg += f' The history[{idx}] is wrong!'
                                    raise IOError(err_msg)
                                history.append((val[0], val[1]))
                            new_sample['history'] = history
                            del history
                        samples.append(new_sample)
                        del new_sample, loaded_sample
            line_idx += 1
            curline = fp.readline()
    return samples


def save_samples(fname: str, samples: List[Dict[str, Union[str, List[Tuple[str, str]]]]]):
    with codecs.open(fname, mode='w', encoding='utf-8') as fp:
        for cur in samples:
            new_sample: Dict[str, Union[str, List[Tuple[str, str]]]] = dict()
            if 'system' in cur:
                new_sample['system'] = cur['system']
            if 'query' in cur:
                new_sample['query'] = cur['query']
            new_sample['response'] = cur['response']
            if 'history' in cur:
                new_sample['history'] = cur['history']
            for k in sorted(list(set(cur.keys()) - {'system', 'query', 'response', 'history'})):
                new_sample[k] = cur[k]
            fp.write(json.dumps(new_sample, ensure_ascii=False) + '\n')
            del new_sample
