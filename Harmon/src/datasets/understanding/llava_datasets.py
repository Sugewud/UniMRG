# Copyright (c) OpenMMLab. All rights reserved.
import json
import logging
import os

import torch
import numpy as np
from einops import rearrange
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk
from mmengine import print_log
from mmengine.config import Config, ConfigDict
from PIL import Image
from torch.utils.data import Dataset

from xtuner.registry import BUILDER
from xtuner.dataset.huggingface import process_hf_dataset
from xtuner.dataset.utils import expand2square


def add_image_token_for_conversations(example):
    has_image_field = 'image' in example and example['image']

    if not has_image_field and 'conversations' in example:
        conversations = example['conversations']

        has_image_token = any('<image>' in str(conv.get('value', '')) for conv in conversations)

        if not has_image_token:
            for i, conv in enumerate(conversations):
                if conv.get('from') == 'human' and 'value' in conv:
                    conversations[i]['value'] = '<image>\n' + conv['value']
                    break

    return example


def load_jsonl(json_file):
    with open(json_file) as f:
        lines = f.readlines()
    data = []
    for line in lines:
        data.append(json.loads(line))
    return data


class MARProcessor:
    def __init__(self, image_size):
        self.image_size = image_size
        self.size = {'height': image_size, 'width': image_size} 

    def __call__(self, image):
        image = expand2square(image, (127, 127, 127))
        image = image.resize(size=(self.image_size, self.image_size))
        image = torch.from_numpy(np.array(image)).to(dtype=torch.float32)
        image = rearrange(image, 'h w c -> c h w')[None]
        image = 2 * (image / 255) - 1
        return {
            'pixel_values': image
        }

    def preprocess(self, image, return_tensors='pt'):
        return self.__call__(image)


# Sibling data dir: ../data/... relative to repository root when training. Override via dataset cache_folder=...
DEFAULT_CACHE_FOLDER = '../data/LLaVA-Instruct-150K/cache_harmon_understanding'


class LLaVADataset(Dataset):

    def __init__(self,
                 image_folder,
                 image_processor=MARProcessor(image_size=512),
                 data_path=None,
                 tokenizer=None,
                 offline_processed_text_folder=None,
                 max_dataset_length=None,
                 dataset_map_fn=None,
                 template_map_fn=None,
                 max_length=2048,
                 pad_image_to_square=False,
                 cache_folder=None):
        super().__init__()
        
        # Default cache path when cache_folder is not provided
        if cache_folder is None:
            cache_folder = DEFAULT_CACHE_FOLDER
        
        assert cache_folder or (data_path and tokenizer)

        # Prefer loading from disk cache when present
        if os.path.exists(cache_folder):
            print_log(f'Loading cached dataset from {cache_folder}', logger='current')
            self.text_data = load_from_disk(cache_folder)
            print_log(f'Loaded {len(self.text_data)} samples from cache.', logger='current')
        else:
            # Build dataset from raw JSON / JSONL when no cache
            if data_path.endswith('.json'):
                json_data = json.load(open(data_path))
            elif data_path.endswith('.jsonl'):
                json_data = load_jsonl(data_path)
            else:
                raise NotImplementedError
            
            for idx in range(len(json_data)):
                if isinstance(json_data[idx].get('id'), int):
                    json_data[idx]['id'] = str(json_data[idx]['id'])
            json_data = DatasetDict({'train': HFDataset.from_list(json_data)})
            print('length of json_data:', len(json_data['train']))
            json_data_processed = json_data.map(add_image_token_for_conversations)
            self.text_data = process_hf_dataset(
                dataset=json_data_processed,
                tokenizer=tokenizer,
                max_length=max_length,
                dataset_map_fn=dataset_map_fn,
                template_map_fn=template_map_fn,
                split='train',
                max_dataset_length=max_dataset_length,
                remove_unused_columns=False,
                pack_to_max_length=False,
                with_image_token=True)
            
            print(f'Processed {len(self.text_data)} samples successfully.')
            
            # Persist processed data to cache_folder
            print(f'Saving processed dataset to {cache_folder}...')
            os.makedirs(cache_folder, exist_ok=True)
            self.text_data.save_to_disk(cache_folder)
            print(f'Dataset cached successfully.')

        self.image_folder = image_folder
        if isinstance(image_processor, dict) or isinstance(
                image_processor, Config) or isinstance(image_processor,
                                                       ConfigDict):
            self.image_processor = BUILDER.build(image_processor)
        else:
            self.image_processor = image_processor
        self.pad_image_to_square = pad_image_to_square

    @property
    def modality_length(self):
        length_list = []
        for data_dict in self.text_data:
            cur_len = len(data_dict['input_ids'])
            if data_dict.get('image', None) is None:
                cur_len = -cur_len
            length_list.append(cur_len)
        return length_list

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, index):
        data_dict = self.text_data[index]
        data_dict['type'] = 'image2text'
        if data_dict.get('image', None) is not None:
            image_file = data_dict['image']
            image = Image.open(os.path.join(self.image_folder,
                                            image_file)).convert('RGB')
            if self.pad_image_to_square:
                image = expand2square(
                    image,
                    tuple(
                        int(x * 255) for x in self.image_processor.image_mean))
            image = self.image_processor.preprocess(
                image, return_tensors='pt')['pixel_values'][0]
            data_dict['pixel_values'] = image
        else:
            if hasattr(self.image_processor, 'crop_size'):
                crop_size = self.image_processor.crop_size
            else:
                crop_size = self.image_processor.size
            data_dict['pixel_values'] = torch.zeros(3, crop_size['height'],
                                                    crop_size['width'])
        return data_dict
