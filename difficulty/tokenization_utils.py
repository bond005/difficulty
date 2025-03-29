from typing import List, Optional, Tuple, Union

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers import BatchEncoding


def tokenize_prompt(prompt: str, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
                    add_eos_token: bool = True, add_labels: bool=True) -> BatchEncoding:
    result = tokenizer(prompt, padding=False, return_tensors=None)
    if (result['input_ids'][-1] != tokenizer.eos_token_id) and add_eos_token:
        result['input_ids'].append(tokenizer.eos_token_id)
        result['attention_mask'].append(1)
    if add_labels:
        result['labels'] = result['input_ids'].copy()
    return result


def prepare_sample(tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
                   user_prompt: str, system_prompt: Optional[str] = None,
                   reference: Optional[str] = None,
                   history: Optional[List[Tuple[str, str]]] = None) -> BatchEncoding:
    if (reference is None) or (len(reference.strip()) == 0):
        tokenized = tokenize_prompt(user_prompt, tokenizer)
    else:
        messages = []
        if system_prompt is not None:
            messages.append({
                'role': 'system',
                'content': system_prompt
            })
        if history is not None:
            if len(history) > 0:
                for turn in history:
                    messages += [
                        {
                            'role': 'user',
                            'content': turn[0]
                        },
                        {
                            'role': 'assistant',
                            'content': turn[1]
                        }
                    ]
        messages.append({
            'role': 'user',
            'content': user_prompt
        })
        text_without_reference = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        number_of_input_tokens = len(tokenize_prompt(text_without_reference, tokenizer)['input_ids']) - 1
        del text_without_reference
        messages.append({
            'role': 'assistant',
            'content': reference
        })
        text_with_reference = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        del messages
        tokenized = tokenize_prompt(text_with_reference, tokenizer)
        tokenized['labels'] = [-100] * number_of_input_tokens + tokenized['labels'][number_of_input_tokens:]
    return tokenized
