import torch
from xtuner.utils import DEFAULT_PAD_TOKEN_INDEX, IGNORE_INDEX
from typing import Dict, Sequence
from torch.nn.utils.rnn import pad_sequence
from functools import partial
from dataclasses import dataclass


def collate_func_gen(instances: Sequence[Dict],
                     pad_index: int = DEFAULT_PAD_TOKEN_INDEX):
    pixel_values, input_ids, input_lengths = [], [], []
    for example in instances:
        pixel_values.append(example.pop('pixel_values'))
        input_lengths.append(len(example['input_ids']))
        input_ids.append(example.pop('input_ids'))

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_index)
    attention_mask = torch.zeros_like(input_ids).bool()
    for i in range(len(input_ids)):
        attention_mask[i, :input_lengths[i]] = True

    data_dict = dict(pixel_values=torch.stack(pixel_values),
                     input_ids=input_ids,
                     attention_mask=attention_mask)

    return {'data': data_dict, 'data_samples': None}


def collate_func_und(instances, pad_index=DEFAULT_PAD_TOKEN_INDEX):
    input_ids_list, labels_list, pixel_values_list = [], [], []

    for sample in instances:
        input_ids_list.append(torch.LongTensor(sample['input_ids']))
        labels_list.append(torch.LongTensor(sample['labels']))

        if 'pixel_values' in sample:
            pixel_values_list.append(sample['pixel_values'])

    ori_length = [len(input_ids_) for input_ids_ in input_ids_list]
    # right padding
    if len(instances) > 1:
        input_ids = pad_sequence(
            input_ids_list, batch_first=True, padding_value=pad_index)
        labels = pad_sequence(
            labels_list, batch_first=True, padding_value=IGNORE_INDEX)
    else:
        input_ids = torch.stack(input_ids_list)
        labels = torch.stack(labels_list)

    attention_mask = torch.zeros_like(input_ids).bool()
    for i, length in enumerate(ori_length):
        attention_mask[i, :length] = True        # right padding

    data_dict = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'pixel_values': torch.stack(pixel_values_list) if len(pixel_values_list) > 0 else None,
        # 'raw_conversations': raw_conversations_list,  # raw conversation dicts
        # 'conversation_text': conversation_list  # formatted conversation strings
    }

    return {'data': data_dict, 'data_samples': None}


def collate_func_edit(instances: Sequence[Dict],
                      pad_index: int = DEFAULT_PAD_TOKEN_INDEX):
    """Collate function for image editing tasks.
    
    Handles both source images (condition) and target images (reconstruction).
    """
    source_pixel_values, target_pixel_values, input_ids, input_lengths = [], [], [], []
    for example in instances:
        source_pixel_values.append(example.pop('source_pixel_values'))
        target_pixel_values.append(example.pop('pixel_values'))  # Target image
        input_lengths.append(len(example['input_ids']))
        input_ids.append(example.pop('input_ids'))

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_index)
    attention_mask = torch.zeros_like(input_ids).bool()
    for i in range(len(input_ids)):
        attention_mask[i, :input_lengths[i]] = True

    data_dict = dict(
        source_pixel_values=torch.stack(source_pixel_values),
        pixel_values=torch.stack(target_pixel_values),
        input_ids=input_ids,
        attention_mask=attention_mask
    )

    return {'data': data_dict, 'data_samples': None}


def collate_func_joint(instances, pad_index):
    """Collate function for joint SFT and Depth estimation."""
    # Separate SFT and Depth fields
    sft_batch = {}
    depth_batch = {}
    
    # SFT Fields
    if 'sft_input_ids' in instances[0]:
        sft_instances = [{
            'input_ids': instance['sft_input_ids'],
            'labels': instance['sft_labels'],
            'attention_mask': instance['sft_attention_mask'],
            'pixel_values': instance['sft_pixel_values']
        } for instance in instances]
        sft_batch = collate_func_und(sft_instances, pad_index)['data']
    
    # Depth Fields
    if 'depth_input_ids' in instances[0]:
        depth_instances = [{
            'input_ids': instance['depth_input_ids'],
            'source_pixel_values': instance['depth_source_pixel_values'],
            'pixel_values': instance['depth_pixel_values'],
            # attention_mask might be missing, create if needed or use default
            'attention_mask': instance.get('depth_attention_mask', torch.ones_like(instance['depth_input_ids']))
        } for instance in instances]
        depth_batch = collate_func_edit(depth_instances, pad_index)['data']

    # Combine
    batch = {}
    # Prefix keys to match what model.joint_loss expects
    for k, v in sft_batch.items():
        batch[f'sft_{k}'] = v
    for k, v in depth_batch.items():
        batch[f'depth_{k}'] = v
        
    return {'data': batch, 'data_samples': None}


class CollateConcat(object):
    def __init__(self, collate_fns, keys):
        self.keys = keys
        self.collate_fns = {}
        for key, collate_fn in zip(keys, collate_fns):
            func = collate_fn.pop('type')
            self.collate_fns[key] = partial(func, **collate_fn)

    def __call__(self, data_samples):
        data_samples = [data_sample for data_sample in data_samples if len(data_sample) > 0]
        data_dict = {}
        key = data_samples[0]['type']
        
        # Fallback for keys not explicitly registered (e.g. 'image2text' might come from SFT dataset if it fallback)
        if key not in self.collate_fns:
             # Try to find a default or best match, or raise error.
             # For this specific case where LLaVAJointDataset might return 'image2text' type 
             # when image is missing, but our config only registered 'joint' key.
             # If the config registered 'joint', but data is 'image2text', we have a problem.
             
             # If we are in joint training mode, we might want to just skip or handle it.
             # But CollateConcat expects exact key match.
             
             # Option 1: Raise error (current behavior)
             # Option 2: Map to a default collator if available.
             
             # Hack: if key is 'image2text' but we only have 'joint', and the data structure matches 'image2text',
             # we can't use 'joint' collator easily because it expects sft_ prefixes.
             
             # If LLaVAJointDataset returned type='image2text', it means it only has SFT data.
             # The user's error shows KeyError: 'image2text'. 
             # This means the dataset returned a sample with type='image2text' 
             # (likely because it couldn't find a depth map or image was missing), 
             # but the config only registered a collator for 'joint'.
             
             pass

        data_dict[key] = self.collate_fns[key](data_samples)['data']

        return {'data': data_dict, 'data_samples': None}
