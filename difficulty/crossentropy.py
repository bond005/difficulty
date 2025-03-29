import os
import sys
from typing import List, Optional, Tuple, Union

import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers import PreTrainedModel, GenerationMixin

try:
    from tokenization_utils import prepare_sample
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from tokenization_utils import prepare_sample


def calculate_crossentropy(tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
                           model: Union[PreTrainedModel, GenerationMixin], user_prompt: str,
                           system_prompt: Optional[str] = None, reference: Optional[str] = None,
                           history: Optional[List[Tuple[str, str]]] = None,
                           max_seq_len: Optional[int] = None) -> Union[Tuple[int, float], None]:
    tokenized = prepare_sample(tokenizer, user_prompt, system_prompt, reference, history)
    seq_len =  len(tokenized['input_ids'])
    if max_seq_len is not None:
        if seq_len >= max_seq_len:
            return None
    tokenized = tokenized.to(model.device)
    with torch.no_grad():
        res = model(input_ids=tokenized['input_ids'], attention_mask=tokenized['attention_mask'],
                    labels=tokenized['labels'], return_dict=True)
        loss_value = float(res.loss.float().cpu().numpy().flatten()[0])
    return seq_len, loss_value
