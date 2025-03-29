import os
import sys
from typing import List, Optional, Tuple, Union

import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers import PreTrainedModel, GenerationMixin

try:
    from difficulty.tokenization_utils import prepare_sample
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from difficulty.tokenization_utils import prepare_sample


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
    with torch.no_grad():
        res = model(
            input_ids=torch.tensor(data=[tokenized['input_ids']], dtype=torch.long, device=model.device),
            attention_mask=torch.tensor(data=[tokenized['attention_mask']], dtype=torch.long, device=model.device),
            labels=torch.tensor(data=[tokenized['labels']], dtype=torch.long, device=model.device),
            return_dict=True
        )
        loss_value = float(res.loss.float().cpu().numpy().flatten()[0])
    return seq_len, loss_value
