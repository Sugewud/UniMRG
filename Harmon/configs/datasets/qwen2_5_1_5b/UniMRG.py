from mmengine.config import read_base
from src.datasets.collate_functions import collate_func_und, collate_func_edit, collate_func_gen, CollateConcat
from src.datasets.samplers.multi_source_sampler import FixedBatchMultiSourceSampler
from src.datasets.understanding.llava_datasets import LLaVADataset
from src.datasets.text2image.text2image import LLaVADepthDataset, LLaVAReconstructionDataset, LLaVAMaskDataset
from xtuner.dataset import ConcatDataset

from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory
from xtuner.dataset.utils import expand2square

with read_base():
    from .processors import prompt_template, tokenizer, image_size, pad_index, image_length


# Data under repo-relative ../data/... (run training from repository root, e.g. via train.sh).
data_root = '../data/LLaVA-Instruct-150K/'
data_path = '../data/LLaVA-Instruct-150K/llava_v1_5_mix665k.json'
image_folder = '../data/LLaVA-Instruct-150K/tuning_data'
depth_folder = '../data/LLaVA-Instruct-150K/tuning_data_depth'
mask_folder = '../data/LLaVA-Instruct-150K/tuning_data_mask'
max_length = int(2048 - (336 / 14) ** 2)

# 1. Image-to-Text (SFT) Dataset
sft_dataset = dict(
    type=LLaVADataset,
    data_path=data_path,
    image_folder=image_folder,
    tokenizer=tokenizer,
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=False,
)

# 2. Depth Reconstruction Dataset
depth_dataset = dict(
    type=LLaVADepthDataset,
    data_path=data_path,        # Use same JSON to get image list
    image_folder=image_folder,
    depth_folder=depth_folder,
    image_size=image_size,
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    crop_image=False,
    max_length=128,             # Short prompt for depth
    unconditional=0.0,
    max_samples=None,
)

# 3. Pixel Reconstruction Dataset
recon_dataset = dict(
    type=LLaVAReconstructionDataset,
    data_path=data_path,
    image_folder=image_folder,
    image_size=image_size,
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    crop_image=False,
    max_length=128,
    unconditional=0.1,
    max_samples=None,
    use_downscale=False,
)

# 4. Mask Segmentation Dataset
mask_dataset = dict(
    type=LLaVAMaskDataset,
    data_path=data_path,        # Use same JSON to get image list
    image_folder=image_folder,
    mask_folder=mask_folder,
    image_size=image_size,
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    crop_image=False,
    max_length=128,             # Short prompt for mask
    unconditional=0.0,
    max_samples=None,
)

# Combine Datasets
dataset = dict(
    type=ConcatDataset,
    datasets=[sft_dataset, depth_dataset, recon_dataset, mask_dataset],
)

group_keys = ['image2text', 'depth', 'recon', 'mask']
repeat = [1, 1, 1, 1]  # Repeat 1 for all
batch_size = 16  # Adjust based on GPU memory

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
    prefetch_factor=1,
    persistent_workers=False,
    pin_memory=True,
    dataset=dataset,
    sampler=dict(type=FixedBatchMultiSourceSampler,
                 repeat=repeat,
                 batch_size=batch_size,    # fixed batch size for all sources
                 shuffle=True),
    collate_fn=dict(type=CollateConcat,
                    collate_fns=[
                        dict(type=collate_func_und, pad_index=pad_index),
                        dict(type=collate_func_edit, pad_index=pad_index),
                        dict(type=collate_func_gen, pad_index=pad_index),
                        dict(type=collate_func_edit, pad_index=pad_index),  # Mask uses same collate as edit
                    ],
                    keys=group_keys
                    )
)
