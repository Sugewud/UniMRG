from torch.utils.data import Dataset
from PIL import Image
import os
import io
import json
import random
import torch
import numpy as np
import logging
from einops import rearrange
from xtuner.registry import BUILDER
from src.datasets.utils import crop2square
from glob import glob

# Imports for LLaVAJointDataset
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk
from mmengine import print_log
from mmengine.config import Config, ConfigDict
from xtuner.dataset.huggingface import process_hf_dataset
from xtuner.dataset.utils import expand2square
from src.datasets.understanding.llava_datasets import add_image_token_for_conversations, load_jsonl, MARProcessor


class Text2ImageDataset(Dataset):
    def __init__(self,
                 data_path,
                 local_folder,
                 image_size,
                 unconditional=0.1,
                 tokenizer=None,
                 prompt_template=None,
                 max_length=1024,
                 crop_image=True,
                 cap_source='caption',
                 ):
        super().__init__()
        self.data_path = data_path
        self._load_data(data_path)
        self.unconditional = unconditional
        self.local_folder = local_folder
        self.cap_source = cap_source

        self.image_size = image_size

        self.tokenizer = BUILDER.build(tokenizer)
        self.prompt_template = prompt_template
        self.max_length = max_length
        self.crop_image = crop_image

    def _load_data(self, data_path):
        with open(data_path, 'r') as f:
            self.data_list = json.load(f)

        print(f"Load {len(self.data_list)} data samples from {data_path}", flush=True)

    def __len__(self):
        return len(self.data_list)

    def _read_image(self, image_file):
        image = Image.open(os.path.join(self.local_folder, image_file))
        assert image.width > 8 and image.height > 8, f"Image: {image.size}"
        assert image.width / image.height > 0.1, f"Image: {image.size}"
        assert image.width / image.height < 10, f"Image: {image.size}"
        return image

    def _process_text(self, text):
        if random.uniform(0, 1) < self.unconditional:
            prompt = "Generate an image."
        else:
            prompt = f"Generate an image: {text.strip()}"
        prompt = self.prompt_template['INSTRUCTION'].format(input=prompt)
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True, return_tensors='pt')[0]

        return dict(input_ids=input_ids[:self.max_length])

    def _process_image(self, image):
        data = dict()

        if self.crop_image:
            image = crop2square(image)
        else:
            target_size = max(image.size)
            image = image.resize(size=(target_size, target_size))

        image = image.resize(size=(self.image_size, self.image_size))
        pixel_values = torch.from_numpy(np.array(image)).float()
        pixel_values = pixel_values / 255
        pixel_values = 2 * pixel_values - 1
        pixel_values = rearrange(pixel_values, 'h w c -> c h w')

        data.update(pixel_values=pixel_values)

        return data

    def _retry(self):
        return self.__getitem__(random.choice(range(self.__len__())))

    def __getitem__(self, idx):
        try:
            data_sample = self.data_list[idx]
            image = self._read_image(data_sample['image']).convert('RGB')

            caption = data_sample[self.cap_source]
            data = self._process_image(image)
            data.update(self._process_text(caption))
            data.update(type='text2image')

            return data

        except Exception as e:
            print(f"Error when reading {self.data_path}:{self.data_list[idx]}: {e}", flush=True)
            return self._retry()


class LargeText2ImageDataset(Text2ImageDataset):
    # self.data_list only contains paths of images and captions

    def __init__(self, cap_folder=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cap_folder = self.local_folder if cap_folder is None else cap_folder

    def _load_data(self, data_path):      # image path and annotation path are saved in a json file
        if data_path.endswith(".json"):
            with open(data_path, 'r') as f:
                self.data_list = json.load(f)
        else:
            self.data_list = []
            json_files = glob(f'{data_path}/*.json')
            for json_file in json_files:
                with open(json_file, 'r') as f:
                    self.data_list += json.load(f)

        print(f"Load {len(self.data_list)} data samples from {data_path}", flush=True)

    def __getitem__(self, idx):
        try:
            data_sample = self.data_list[idx]
            image = self._read_image(data_sample['image']).convert('RGB')
            with open(f"{self.cap_folder}/{data_sample['annotation']}", 'r') as f:
                caption = json.load(f)[self.cap_source]
            data = self._process_image(image)
            data.update(self._process_text(caption))
            data.update(type='text2image')
            return data

        except Exception as e:
            print(f"Error when reading {self.data_path}:{data_sample}: {e}", flush=True)
            return self._retry()


class BlipO3Dataset(Text2ImageDataset):
    def __init__(self, 
                 data_path="/scratch/2025_06/jixie/BLIP3o-60k/*.tar",
                 cache_dir='/scratch/2025_06/jixie/',
                 *args, **kwargs):
        self.data_path = data_path
        self.cache_dir = cache_dir
        super().__init__(data_path=data_path, *args, **kwargs)

    def _load_data(self, data_path):
        try:
            from datasets import load_dataset
            print(f"Loading dataset from {data_path} with cache_dir {self.cache_dir}")
            data_files = glob(data_path) 
            self.dataset = load_dataset("webdataset", data_files=data_files, cache_dir=self.cache_dir, split="train", num_proc=64)

            print(f"Loaded {len(self.dataset)} samples from {data_path}")
            
            self.data_list = []
            for idx in range(len(self.dataset)):
                self.data_list.append({
                    'idx': idx,
                })
                
        except Exception as e:
            print(f"Error loading dataset: {e}")
            self.data_list = []

        print(f"Load {len(self.data_list)} data samples from {data_path}", flush=True)

    def __getitem__(self, idx):
        try:
            data_sample = self.data_list[idx]
            original_idx = data_sample['idx']
            
            sample = self.dataset[original_idx]
            
            image_data = sample['jpg']
            if isinstance(image_data, dict) and 'bytes' in image_data:
                image = Image.open(io.BytesIO(image_data['bytes'])).convert('RGB')
            elif hasattr(image_data, 'convert'):
                image = image_data.convert('RGB')
            elif isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
            else:
                try:
                    image = Image.fromarray(np.array(image_data)).convert('RGB')
                except:
                    raise TypeError(f"Unknown type: {type(image_data)}")
            
            caption = sample['txt']
            
            data = self._process_image(image)
            data.update(self._process_text(caption))
            data.update(type='text2image')
            return data

        except Exception as e:
            print(f"Error when processing index {idx}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return self._retry()
        
        
class MidJourneyDataset(Text2ImageDataset):
    def __init__(self, 
                 data_path="brivangl/midjourney-v6-llava",
                 cache_dir=None,
                 use_llava=False,
                 *args, **kwargs):
        self.data_path = data_path
        self.cache_dir = cache_dir
        self.use_llava = use_llava
        super().__init__(data_path=data_path, *args, **kwargs)

    def _load_data(self, data_path):
        try:
            from datasets import load_dataset
            print(f"Loading dataset from {data_path} with cache_dir {self.cache_dir}")
            self.dataset = load_dataset(data_path, cache_dir=self.cache_dir)['train']
            print(f"Loaded {len(self.dataset)} samples from {data_path}")
            
            self.data_list = []
            for idx in range(len(self.dataset)):
                self.data_list.append({
                    'idx': idx, 
                })
        except Exception as e:
            print(f"Error loading dataset: {e}")
            self.data_list = []

        print(f"Load {len(self.data_list)} data samples from {data_path}", flush=True)

    def __getitem__(self, idx):
        try:
            data_sample = self.data_list[idx]
            original_idx = data_sample['idx']
            
            sample = self.dataset[original_idx]
            
            image_data = sample['image']
            if isinstance(image_data, dict) and 'bytes' in image_data:
                image = Image.open(io.BytesIO(image_data['bytes'])).convert('RGB')
            elif hasattr(image_data, 'convert'):
                image = image_data.convert('RGB')
            elif isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
            else:
                try:
                    image = Image.fromarray(np.array(image_data)).convert('RGB')
                except:
                    raise TypeError(f"Unknown type: {type(image_data)}")
            
            if self.use_llava:
                caption = sample['llava']
            else:
                caption = sample['prompt']
            
            data = self._process_image(image)
            data.update(self._process_text(caption))
            data.update(type='text2image')
            return data

        except Exception as e:
            print(f"Error when processing index {idx}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return self._retry()

class ReconstructionDataset(Text2ImageDataset):
    def __init__(self,
                 data_path,
                 image_size,
                 unconditional=0.1,
                 tokenizer=None,
                 prompt_template=None,
                 max_length=1024,
                 crop_image=False,
                 cap_source='caption',
                 max_samples=None,
                 use_downscale=False,
                 cache_dir='/scratch/2025_06/jixie/journeydb'):
        
        self.data_path = data_path
        self.unconditional = unconditional
        self.local_folder = None
        self.cap_source = cap_source
        self.image_size = image_size
        self.tokenizer = BUILDER.build(tokenizer)
        self.prompt_template = prompt_template
        self.max_length = max_length
        self.crop_image = crop_image
        self.max_samples = max_samples
        self.use_downscale = use_downscale
        self.cache_dir = cache_dir
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self._load_data(data_path)
        from src.datasets.text2image.consts import get_recon_prompt_list
        self.recon_prompts = get_recon_prompt_list()
        
        print(f"Loaded ReconstructionDataset with {len(self.data_list)} samples, {len(self.recon_prompts)} prompts, cache_dir: {self.cache_dir}", flush=True)

    def _extract_tar_if_needed(self, tar_path):
        import tarfile
        import hashlib
        tar_hash = hashlib.md5(tar_path.encode()).hexdigest()
        extract_dir = os.path.join(self.cache_dir, tar_hash)
        
        lock_file = os.path.join(extract_dir, '.extraction_complete')
        
        if os.path.exists(lock_file):
            print(f"Using cached extraction for {tar_path} in {extract_dir}", flush=True)
            return extract_dir
        
        print(f"Extracting {tar_path} to {extract_dir}...", flush=True)
        os.makedirs(extract_dir, exist_ok=True)
        
        try:
            with tarfile.open(tar_path, 'r') as tar:
                tar.extractall(path=extract_dir)
            
            with open(lock_file, 'w') as f:
                f.write(f"Extracted from {tar_path} at {os.path.getmtime(tar_path)}")
                
            print(f"Extraction complete: {tar_path} -> {extract_dir}", flush=True)
            return extract_dir
        except Exception as e:
            print(f"Error extracting {tar_path}: {e}", flush=True)
            raise
    
    def _load_data(self, data_path):
        import tarfile
        import glob
        
        self.tar_files = glob.glob(os.path.expanduser(data_path.replace('{', '[').replace('}', ']')))
        self.data_list = []
        self.image_cache_paths = {}
        
        for tar_idx, tar_path in enumerate(self.tar_files):
            try:
                extract_dir = self._extract_tar_if_needed(tar_path)
                
                with tarfile.open(tar_path, 'r') as tar:
                    for member in tar.getmembers():
                        if member.isfile() and member.name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                            file_name = member.name
                            cache_path = os.path.join(extract_dir, file_name)
                            self.data_list.append({'image': file_name, 'tar_idx': tar_idx})
                            self.image_cache_paths[file_name] = cache_path
                        
                        if self.max_samples and len(self.data_list) >= self.max_samples:
                            break
                    
                    if self.max_samples and len(self.data_list) >= self.max_samples:
                        break
                    
            except Exception as e:
                print(f"Error loading tar file {tar_path}: {e}", flush=True)
                
        print(f"Loaded {len(self.data_list)} images from {len(self.tar_files)} tar files: {self.tar_files}", flush=True)

        if len(self.data_list) == 0:
            raise RuntimeError(f"No valid images found in tar archives: {data_path}")

    def _read_image(self, image_file):
        if image_file not in self.image_cache_paths:
            raise ValueError(f"Image file {image_file} not found in cache")
            
        cache_path = self.image_cache_paths[image_file]
        
        try:
            image = Image.open(cache_path)
            assert image.width > 8 and image.height > 8, f"Image too small: {image.size}"
            assert image.width / image.height > 0.1, f"Image aspect ratio too extreme: {image.size}"
            assert image.width / image.height < 10, f"Image aspect ratio too extreme: {image.size}"
            
            return image
        except Exception as e:
            raise RuntimeError(f"Error reading image from cache path {cache_path}: {e}")

    def _process_text(self, text):
        prompt = random.choice(self.recon_prompts)
        if random.uniform(0, 1) < self.unconditional:
            final_prompt = "Generate an image."
        else:
            final_prompt = f"\n{prompt}"
            
        final_prompt = self.prompt_template['INSTRUCTION'].format(input=final_prompt)
        input_ids = self.tokenizer.encode(final_prompt, add_special_tokens=True, return_tensors='pt')[0]
        # print(f"Prompt: {final_prompt}", flush=True)
        input_ids = torch.cat([
            input_ids[:3],
            torch.tensor([-200], dtype=torch.long),
            input_ids[3:],
        ], dim=0)
        return dict(input_ids=input_ids[:self.max_length])

    def __getitem__(self, idx):
        try:
            data_sample = self.data_list[idx]
            image = self._read_image(data_sample['image']).convert('RGB')
            
            if self.use_downscale:
                image = image.resize(size=(self.image_size // 2, self.image_size // 2))
                image = image.resize(size=(self.image_size, self.image_size))
            
            data = self._process_image(image)
            data.update(self._process_text(""))
            data.update(type='recon')
            
            return data

        except Exception as e:
            print(f"Error when processing index {idx}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return self._retry()

class MidjourneyReconstructionDataset(Text2ImageDataset):
    def __init__(self, 
                 image_size,
                 data_path="brivangl/midjourney-v6-llava",
                 cache_dir=None,
                 unconditional=0.1,
                 tokenizer=None,
                 prompt_template=None,
                 max_length=1024,
                 crop_image=False,
                 cap_source='caption',
                 max_samples=None,
                 use_downscale=False,
                 *args, **kwargs):
        self.data_path = data_path
        self.unconditional = unconditional
        self.local_folder = None
        self.cap_source = cap_source
        self.image_size = image_size
        self.tokenizer = BUILDER.build(tokenizer)
        self.prompt_template = prompt_template
        self.max_length = max_length
        self.crop_image = crop_image
        self.max_samples = max_samples
        self.use_downscale = use_downscale
        self.cache_dir = cache_dir
        from src.datasets.text2image.consts import get_recon_prompt_list
        self.recon_prompts = get_recon_prompt_list()
        self._load_data(data_path)

    def _load_data(self, data_path):
        try:
            from datasets import load_dataset
            print(f"Loading dataset from {data_path} with cache_dir {self.cache_dir}")
            self.dataset = load_dataset(data_path, cache_dir=self.cache_dir)['train']
            print(f"Loaded {len(self.dataset)} samples from {data_path}")
            
            self.data_list = []
            for idx in range(len(self.dataset)):
                self.data_list.append({
                    'idx': idx,
                })
        except Exception as e:
            print(f"Error loading dataset: {e}")
            self.data_list = []

        print(f"Load {len(self.data_list)} data samples from {data_path} for reconstruction", flush=True)
    
    def _process_text(self, text):
        prompt = random.choice(self.recon_prompts)
        
        if random.uniform(0, 1) < self.unconditional:
            final_prompt = "Generate an image."
        else:
            final_prompt = f"\n{prompt}"
            
        final_prompt = self.prompt_template['INSTRUCTION'].format(input=final_prompt)
        input_ids = self.tokenizer.encode(final_prompt, add_special_tokens=True, return_tensors='pt')[0]
        
        input_ids = torch.cat([
            input_ids[:3],
            torch.tensor([-200], dtype=torch.long),
            input_ids[3:],
        ], dim=0)
        
        return dict(input_ids=input_ids[:self.max_length])

    def __getitem__(self, idx):
        try:
            data_sample = self.data_list[idx]
            original_idx = data_sample['idx']
            
            sample = self.dataset[original_idx]
            
            image_data = sample['image']
            if isinstance(image_data, dict) and 'bytes' in image_data:
                image = Image.open(io.BytesIO(image_data['bytes'])).convert('RGB')
            elif hasattr(image_data, 'convert'):
                image = image_data.convert('RGB')
            elif isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
            else:
                try:
                    image = Image.fromarray(np.array(image_data)).convert('RGB')
                except:
                    raise TypeError(f"Unknown type: {type(image_data)}")

            data = self._process_image(image)
            data.update(self._process_text(""))
            data.update(type='recon')
            
            return data

        except Exception as e:
            print(f"Error when processing index {idx}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return self._retry()
        
    def __len__(self):

        return len(self.data_list) if self.data_list else 0


class LLaVADepthDataset(Dataset):
    """Depth estimation dataset using LLaVA images and pre-computed depth maps.
    
    Input: LLaVA image
    Output: Depth map
    Task: Generate depth map from RGB image
    """
    def __init__(self, 
                 data_path,
                 image_folder,
                 depth_folder,
                 image_size=512,
                 unconditional=0.0,
                 tokenizer=None,
                 prompt_template=None,
                 max_length=1024,
                 crop_image=False,
                 max_samples=None,
                 *args, **kwargs):
        self.data_path = data_path
        self.image_folder = image_folder
        self.depth_folder = depth_folder
        self.unconditional = unconditional
        self.image_size = image_size
        self.tokenizer = BUILDER.build(tokenizer)
        self.prompt_template = prompt_template
        self.max_length = max_length
        self.crop_image = crop_image
        self.max_samples = max_samples
        self._load_data(data_path)

    def _load_data(self, data_path):
        import json
        with open(data_path, 'r') as f:
            self.raw_data = json.load(f)
            
        print(f"Loaded {len(self.raw_data)} samples from {data_path}", flush=True)
        
        self.data_list = []
        for item in self.raw_data:
            if 'image' in item:
                self.data_list.append(item['image'])
        
        if self.max_samples:
            self.data_list = self.data_list[:self.max_samples]
            
        print(f"Found {len(self.data_list)} images for depth reconstruction", flush=True)

    def _process_image(self, image):
        """Process RGB input image."""
        if self.crop_image:
            image = crop2square(image)
        else:
            target_size = max(image.size)
            image = image.resize(size=(target_size, target_size))
        
        image = image.resize(size=(self.image_size, self.image_size))
        pixel_values = torch.from_numpy(np.array(image)).float()
        pixel_values = pixel_values / 255
        pixel_values = 2 * pixel_values - 1
        pixel_values = rearrange(pixel_values, 'h w c -> c h w')
        
        return pixel_values

    def _process_depth(self, depth_img):
        """Process depth map image as target output.
        
        Assumes depth map is a grayscale image.
        """
        if self.crop_image:
            depth_img = crop2square(depth_img)
        else:
            target_size = max(depth_img.size)
            depth_img = depth_img.resize(size=(target_size, target_size))
            
        depth_img = depth_img.resize(size=(self.image_size, self.image_size), resample=Image.BILINEAR)
        
        # Convert to numpy and normalize
        depth_array = np.array(depth_img).astype(np.float32)
        
        # If it's single channel
        if len(depth_array.shape) == 2:
            pass
        elif len(depth_array.shape) == 3:
            depth_array = depth_array[:, :, 0] # Take first channel if it's RGB/RGBA but actually grayscale
            
        # Normalize to [0, 1]
        d_min = depth_array.min()
        d_max = depth_array.max()
        if d_max - d_min > 1e-6:
            depth_normalized = (depth_array - d_min) / (d_max - d_min)
        else:
            depth_normalized = np.zeros_like(depth_array)
        
        # Convert to [-1, 1]
        depth_normalized = 2 * depth_normalized - 1
        
        # Convert to 3-channel (repeat the same depth values)
        depth_3ch = np.stack([depth_normalized, depth_normalized, depth_normalized], axis=0)
        depth_tensor = torch.from_numpy(depth_3ch).float()
        
        return depth_tensor

    def _process_text(self):
        """Process text prompt for depth estimation."""
        depth_prompts = [
            "Generate the depth map of this image.",
            "Produce a depth estimation for this image.",
            "Create a depth map showing the distance of objects in this image.",
            "Estimate the depth information from this image.",
            "Generate a depth prediction for this image.",
        ]
        
        if random.uniform(0, 1) < self.unconditional:
            final_prompt = "Generate an image."
        else:
            prompt = random.choice(depth_prompts)
            final_prompt = f"\n{prompt}"
        
        final_prompt = self.prompt_template['INSTRUCTION'].format(input=final_prompt)
        input_ids = self.tokenizer.encode(final_prompt, add_special_tokens=True, return_tensors='pt')[0]
        
        # Insert IMAGE_TOKEN_INDEX placeholder for source image
        input_ids = torch.cat([
            input_ids[:3],
            torch.tensor([-200], dtype=torch.long),
            input_ids[3:],
        ], dim=0)
        
        return dict(input_ids=input_ids[:self.max_length])

    def _retry(self):
        return self.__getitem__(random.choice(range(len(self.data_list))))

    def __getitem__(self, idx):
        try:
            image_file = self.data_list[idx]
            
            # Load RGB image
            rgb_path = os.path.join(self.image_folder, image_file)
            rgb_img = Image.open(rgb_path).convert('RGB')
            
            # Load Depth image
            # Try same extension first, then others if needed. 
            # Assuming exact same relative path structure.
            depth_path = os.path.join(self.depth_folder, image_file)
            
            # If extension might differ (e.g. jpg -> png), check existence
            if not os.path.exists(depth_path):
                base, ext = os.path.splitext(depth_path)
                if os.path.exists(base + '.png'):
                    depth_path = base + '.png'
                elif os.path.exists(base + '.jpg'):
                    depth_path = base + '.jpg'
                # Else fail and retry
            
            depth_img = Image.open(depth_path) # Keep original mode first
            
            # Process input RGB image
            source_pixel_values = self._process_image(rgb_img)
            
            # Process target depth map
            target_pixel_values = self._process_depth(depth_img)
            
            data = dict(
                source_pixel_values=source_pixel_values,
                pixel_values=target_pixel_values,  # Target depth map
            )
            data.update(self._process_text())
            data.update(type='depth')  # Use 'depth' type for depth reconstruction
            
            return data

        except Exception as e:
            print(f"Error when processing index {idx}: {e}", flush=True)
            # import traceback
            # traceback.print_exc()
            return self._retry()

    def __len__(self):
        return len(self.data_list)


class LLaVAReconstructionDataset(Text2ImageDataset):
    def __init__(self, 
                 data_path,
                 image_folder,
                 image_size=512,
                 unconditional=0.1,
                 tokenizer=None,
                 prompt_template=None,
                 max_length=1024,
                 crop_image=False,
                 max_samples=None,
                 use_downscale=False,
                 *args, **kwargs):
        self.data_path = data_path
        self.image_folder = image_folder
        self.unconditional = unconditional
        self.image_size = image_size
        self.tokenizer = BUILDER.build(tokenizer)
        self.prompt_template = prompt_template
        self.max_length = max_length
        self.crop_image = crop_image
        self.max_samples = max_samples
        self.use_downscale = use_downscale
        
        from src.datasets.text2image.consts import get_recon_prompt_list
        self.recon_prompts = get_recon_prompt_list()
        
        self._load_data(data_path)

    def _load_data(self, data_path):
        import json
        with open(data_path, 'r') as f:
            self.raw_data = json.load(f)
            
        print(f"Loaded {len(self.raw_data)} samples from {data_path}", flush=True)
        
        self.data_list = []
        for item in self.raw_data:
            if 'image' in item:
                self.data_list.append(item['image'])
        
        if self.max_samples:
            self.data_list = self.data_list[:self.max_samples]
            
        print(f"Found {len(self.data_list)} images for reconstruction", flush=True)

    def _read_image(self, image_file):
        image_path = os.path.join(self.image_folder, image_file)
        try:
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            raise RuntimeError(f"Error reading image {image_path}: {e}")

    def _process_text(self, text):
        prompt = random.choice(self.recon_prompts)
        if random.uniform(0, 1) < self.unconditional:
            final_prompt = "Generate an image."
        else:
            final_prompt = f"\\n{prompt}"
            
        final_prompt = self.prompt_template['INSTRUCTION'].format(input=final_prompt)
        input_ids = self.tokenizer.encode(final_prompt, add_special_tokens=True, return_tensors='pt')[0]
        
        input_ids = torch.cat([
            input_ids[:3],
            torch.tensor([-200], dtype=torch.long),
            input_ids[3:],
        ], dim=0)
        return dict(input_ids=input_ids[:self.max_length])

    def __getitem__(self, idx):
        try:
            image_file = self.data_list[idx]
            image = self._read_image(image_file)
            
            if self.use_downscale:
                image = image.resize(size=(self.image_size // 2, self.image_size // 2))
                image = image.resize(size=(self.image_size, self.image_size))
            
            data = self._process_image(image)
            data.update(self._process_text(""))
            data.update(type='recon')
            
            return data

        except Exception as e:
            print(f"Error when processing index {idx}: {e}", flush=True)
            return self._retry()

    def _retry(self):
         return self.__getitem__(random.choice(range(len(self.data_list))))

    def __len__(self):
        return len(self.data_list)


class LLaVASketchDataset(Dataset):
    """Sketch generation dataset using LLaVA images and pre-computed sketches.
    
    Input: LLaVA image
    Output: Sketch
    Task: Generate sketch from RGB image
    """
    def __init__(self, 
                 data_path,
                 image_folder,
                 sketch_folder,
                 image_size=512,
                 unconditional=0.0,
                 tokenizer=None,
                 prompt_template=None,
                 max_length=1024,
                 crop_image=False,
                 max_samples=None,
                 *args, **kwargs):
        self.data_path = data_path
        self.image_folder = image_folder
        self.sketch_folder = sketch_folder
        self.unconditional = unconditional
        self.image_size = image_size
        self.tokenizer = BUILDER.build(tokenizer)
        self.prompt_template = prompt_template
        self.max_length = max_length
        self.crop_image = crop_image
        self.max_samples = max_samples
        self._load_data(data_path)

    def _load_data(self, data_path):
        import json
        with open(data_path, 'r') as f:
            self.raw_data = json.load(f)
            
        print(f"Loaded {len(self.raw_data)} samples from {data_path}", flush=True)
        
        self.data_list = []
        for item in self.raw_data:
            if 'image' in item:
                self.data_list.append(item['image'])
        
        if self.max_samples:
            self.data_list = self.data_list[:self.max_samples]
            
        print(f"Found {len(self.data_list)} images for sketch generation", flush=True)

    def _process_image(self, image):
        """Process RGB input image."""
        if self.crop_image:
            image = crop2square(image)
        else:
            target_size = max(image.size)
            image = image.resize(size=(target_size, target_size))
        
        image = image.resize(size=(self.image_size, self.image_size))
        pixel_values = torch.from_numpy(np.array(image)).float()
        pixel_values = pixel_values / 255
        pixel_values = 2 * pixel_values - 1
        pixel_values = rearrange(pixel_values, 'h w c -> c h w')
        
        return pixel_values

    def _process_sketch(self, sketch_img):
        """Process sketch image as target output.
        
        Assumes sketch is a grayscale or RGB image.
        """
        if self.crop_image:
            sketch_img = crop2square(sketch_img)
        else:
            target_size = max(sketch_img.size)
            sketch_img = sketch_img.resize(size=(target_size, target_size))
            
        sketch_img = sketch_img.resize(size=(self.image_size, self.image_size), resample=Image.BILINEAR)
        
        # Convert to numpy
        sketch_array = np.array(sketch_img).astype(np.float32)
        
        # If it's single channel (grayscale)
        if len(sketch_array.shape) == 2:
            # Normalize to [0, 1]
            sketch_normalized = sketch_array / 255.0
            # Convert to [-1, 1]
            sketch_normalized = 2 * sketch_normalized - 1
            # Convert to 3-channel (repeat the same values)
            sketch_3ch = np.stack([sketch_normalized, sketch_normalized, sketch_normalized], axis=0)
        elif len(sketch_array.shape) == 3:
            # Already RGB, normalize
            sketch_normalized = sketch_array / 255.0
            sketch_normalized = 2 * sketch_normalized - 1
            sketch_3ch = rearrange(sketch_normalized, 'h w c -> c h w')
        else:
            raise ValueError(f"Unexpected sketch shape: {sketch_array.shape}")
        
        sketch_tensor = torch.from_numpy(sketch_3ch).float()
        
        return sketch_tensor

    def _process_text(self):
        """Process text prompt for sketch generation."""
        from src.datasets.text2image.consts import get_sketch_prompt_list
        sketch_prompts = get_sketch_prompt_list()
        
        if random.uniform(0, 1) < self.unconditional:
            final_prompt = "Generate an image."
        else:
            prompt = random.choice(sketch_prompts)
            final_prompt = f"\n{prompt}"
        
        final_prompt = self.prompt_template['INSTRUCTION'].format(input=final_prompt)
        input_ids = self.tokenizer.encode(final_prompt, add_special_tokens=True, return_tensors='pt')[0]
        
        # Insert IMAGE_TOKEN_INDEX placeholder for source image
        input_ids = torch.cat([
            input_ids[:3],
            torch.tensor([-200], dtype=torch.long),
            input_ids[3:],
        ], dim=0)
        
        return dict(input_ids=input_ids[:self.max_length])

    def _retry(self):
        return self.__getitem__(random.choice(range(len(self.data_list))))

    def __getitem__(self, idx):
        try:
            image_file = self.data_list[idx]
            
            # Load RGB image
            rgb_path = os.path.join(self.image_folder, image_file)
            rgb_img = Image.open(rgb_path).convert('RGB')
            
            # Load Sketch image
            sketch_path = os.path.join(self.sketch_folder, image_file)
            
            # If extension might differ (e.g. jpg -> png), check existence
            if not os.path.exists(sketch_path):
                base, ext = os.path.splitext(sketch_path)
                if os.path.exists(base + '.png'):
                    sketch_path = base + '.png'
                elif os.path.exists(base + '.jpg'):
                    sketch_path = base + '.jpg'
            
            sketch_img = Image.open(sketch_path)
            
            # Process input RGB image
            source_pixel_values = self._process_image(rgb_img)
            
            # Process target sketch
            target_pixel_values = self._process_sketch(sketch_img)
            
            data = dict(
                source_pixel_values=source_pixel_values,
                pixel_values=target_pixel_values,  # Target sketch
            )
            data.update(self._process_text())
            data.update(type='sketch')  # Use 'sketch' type
            
            return data

        except Exception as e:
            print(f"Error when processing index {idx}: {e}", flush=True)
            return self._retry()

    def __len__(self):
        return len(self.data_list)


class LLaVAMaskDataset(Dataset):
    """Mask segmentation dataset using LLaVA images and pre-computed masks from SAM.
    
    Input: LLaVA image
    Output: Segmentation mask
    Task: Generate segmentation mask from RGB image
    """
    def __init__(self, 
                 data_path,
                 image_folder,
                 mask_folder,
                 image_size=512,
                 unconditional=0.0,
                 tokenizer=None,
                 prompt_template=None,
                 max_length=1024,
                 crop_image=False,
                 max_samples=None,
                 *args, **kwargs):
        self.data_path = data_path
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.unconditional = unconditional
        self.image_size = image_size
        self.tokenizer = BUILDER.build(tokenizer)
        self.prompt_template = prompt_template
        self.max_length = max_length
        self.crop_image = crop_image
        self.max_samples = max_samples
        self._load_data(data_path)

    def _load_data(self, data_path):
        import json
        with open(data_path, 'r') as f:
            self.raw_data = json.load(f)
            
        print(f"Loaded {len(self.raw_data)} samples from {data_path}", flush=True)
        
        self.data_list = []
        for item in self.raw_data:
            if 'image' in item:
                self.data_list.append(item['image'])
        
        if self.max_samples:
            self.data_list = self.data_list[:self.max_samples]
            
        print(f"Found {len(self.data_list)} images for mask generation", flush=True)

    def _process_image(self, image):
        """Process RGB input image."""
        if self.crop_image:
            image = crop2square(image)
        else:
            target_size = max(image.size)
            image = image.resize(size=(target_size, target_size))
        
        image = image.resize(size=(self.image_size, self.image_size))
        pixel_values = torch.from_numpy(np.array(image)).float()
        pixel_values = pixel_values / 255
        pixel_values = 2 * pixel_values - 1
        pixel_values = rearrange(pixel_values, 'h w c -> c h w')
        
        return pixel_values

    def _process_mask(self, mask_img):
        """Process mask image as target output.
        
        Handles SAM output masks which can be RGB images with colored regions.
        """
        if self.crop_image:
            mask_img = crop2square(mask_img)
        else:
            target_size = max(mask_img.size)
            mask_img = mask_img.resize(size=(target_size, target_size))
            
        # Use NEAREST for mask to preserve hard boundaries
        mask_img = mask_img.resize(size=(self.image_size, self.image_size), resample=Image.NEAREST)
        
        # Convert to numpy
        mask_array = np.array(mask_img).astype(np.float32)
        
        # If it's single channel (grayscale/binary mask)
        if len(mask_array.shape) == 2:
            # Normalize to [0, 1]
            mask_normalized = mask_array / 255.0
            # Convert to [-1, 1]
            mask_normalized = 2 * mask_normalized - 1
            # Convert to 3-channel (repeat the same values)
            mask_3ch = np.stack([mask_normalized, mask_normalized, mask_normalized], axis=0)
        elif len(mask_array.shape) == 3:
            # Already RGB (colored mask), normalize
            mask_normalized = mask_array / 255.0
            mask_normalized = 2 * mask_normalized - 1
            mask_3ch = rearrange(mask_normalized, 'h w c -> c h w')
        else:
            raise ValueError(f"Unexpected mask shape: {mask_array.shape}")
        
        mask_tensor = torch.from_numpy(mask_3ch).float()
        
        return mask_tensor

    def _process_text(self):
        """Process text prompt for mask generation."""
        from src.datasets.text2image.consts import get_mask_prompt_list
        mask_prompts = get_mask_prompt_list()
        
        if random.uniform(0, 1) < self.unconditional:
            final_prompt = "Generate an image."
        else:
            prompt = random.choice(mask_prompts)
            final_prompt = f"\n{prompt}"
        
        final_prompt = self.prompt_template['INSTRUCTION'].format(input=final_prompt)
        input_ids = self.tokenizer.encode(final_prompt, add_special_tokens=True, return_tensors='pt')[0]
        
        # Insert IMAGE_TOKEN_INDEX placeholder for source image
        input_ids = torch.cat([
            input_ids[:3],
            torch.tensor([-200], dtype=torch.long),
            input_ids[3:],
        ], dim=0)
        
        return dict(input_ids=input_ids[:self.max_length])

    def _retry(self):
        return self.__getitem__(random.choice(range(len(self.data_list))))

    def __getitem__(self, idx):
        try:
            image_file = self.data_list[idx]
            
            # Load RGB image
            rgb_path = os.path.join(self.image_folder, image_file)
            rgb_img = Image.open(rgb_path).convert('RGB')
            
            # Load Mask image
            mask_path = os.path.join(self.mask_folder, image_file)
            
            # If extension might differ (e.g. jpg -> png), check existence
            if not os.path.exists(mask_path):
                base, ext = os.path.splitext(mask_path)
                if os.path.exists(base + '.png'):
                    mask_path = base + '.png'
                elif os.path.exists(base + '.jpg'):
                    mask_path = base + '.jpg'
            
            mask_img = Image.open(mask_path)
            
            # Process input RGB image
            source_pixel_values = self._process_image(rgb_img)
            
            # Process target mask
            target_pixel_values = self._process_mask(mask_img)
            
            data = dict(
                source_pixel_values=source_pixel_values,
                pixel_values=target_pixel_values,  # Target mask
            )
            data.update(self._process_text())
            data.update(type='mask')  # Use 'mask' type
            
            return data

        except Exception as e:
            print(f"Error when processing index {idx}: {e}", flush=True)
            return self._retry()

    def __len__(self):
        return len(self.data_list)


class LLaVAJointDataset(Dataset):
    """Joint dataset for SFT (Image-to-Text) and Depth Estimation.
    
    Each sample contains data for both tasks:
    1. SFT: RGB Image + Conversation (Input IDs, Labels)
    2. Depth: RGB Image (Source) + Depth Map (Target)
    """
    def __init__(self, 
                 data_path,
                 image_folder,
                 depth_folder,
                 image_processor=None,
                 tokenizer=None,
                 template_map_fn=None,
                 max_length=2048,
                 pad_image_to_square=False,
                 depth_image_size=512,
                 depth_unconditional=0.0,
                 depth_prompt_template=None,
                 depth_max_length=128,
                 crop_image=False,
                 max_samples=None,
                 offline_processed_text_folder=None,
                 dataset_map_fn=None,
                 max_dataset_length=None,
                 *args, **kwargs):
        super().__init__()
        
        # --- SFT Initialization (Logic from LLaVADataset) ---
        self.image_folder = image_folder
        self.pad_image_to_square = pad_image_to_square
        
        # Initialize Image Processor
        if image_processor is None:
            self.image_processor = MARProcessor(image_size=512)
        elif isinstance(image_processor, dict) or isinstance(image_processor, Config) or isinstance(image_processor, ConfigDict):
            self.image_processor = BUILDER.build(image_processor)
        else:
            self.image_processor = image_processor

        # Initialize Tokenizer (Build if Config)
        if isinstance(tokenizer, dict) or isinstance(tokenizer, Config) or isinstance(tokenizer, ConfigDict):
            self.tokenizer = BUILDER.build(tokenizer)
        else:
            self.tokenizer = tokenizer

        # Initialize Text Data (SFT)
        assert offline_processed_text_folder or (data_path and self.tokenizer)
        if offline_processed_text_folder and data_path:
            print_log('Both `offline_processed_text_folder` and `data_path` are set...', logger='current', level=logging.WARNING)

        if offline_processed_text_folder is not None:
            self.text_data = load_from_disk(offline_processed_text_folder)
        else:
            if data_path.endswith('.json'):
                json_data = json.load(open(data_path))
            elif data_path.endswith('.jsonl'):
                json_data = load_jsonl(data_path)
            else:
                raise NotImplementedError
            
            for idx in range(len(json_data)):
                if isinstance(json_data[idx]['id'], int):
                    json_data[idx]['id'] = str(json_data[idx]['id'])
            
            # Store raw data for Depth task (need to access 'image' field directly)
            self.raw_json_data = json_data
            
            json_data_hf = DatasetDict({'train': HFDataset.from_list(json_data)})
            print('length of json_data:', len(json_data_hf['train']))
            json_data_processed = json_data_hf.map(add_image_token_for_conversations)
            self.text_data = process_hf_dataset(
                dataset=json_data_processed,
                tokenizer=self.tokenizer,
                max_length=max_length,
                dataset_map_fn=dataset_map_fn,
                template_map_fn=template_map_fn,
                split='train',
                max_dataset_length=max_dataset_length,
                remove_unused_columns=False,
                pack_to_max_length=False,
                with_image_token=True)

        # --- Depth Initialization (Logic from LLaVADepthDataset) ---
        self.depth_folder = depth_folder
        self.depth_unconditional = depth_unconditional
        self.depth_image_size = depth_image_size
        # self.tokenizer is already initialized and shared
        self.depth_prompt_template = depth_prompt_template
        self.depth_max_length = depth_max_length
        self.crop_image = crop_image
        
        # Ensure alignment: The processed text_data should align with raw_json_data indices
        # process_hf_dataset might filter or reorder? Usually it keeps order if not packed.
        # We assume 1-to-1 mapping for simplicity.
        # If max_samples is set, we truncate.
        if max_samples:
            self.text_data = self.text_data.select(range(min(len(self.text_data), max_samples)))

    def _process_depth_image(self, image):
        """Process RGB input image for Depth task."""
        if self.crop_image:
            image = crop2square(image)
        else:
            target_size = max(image.size)
            image = image.resize(size=(target_size, target_size))
        
        image = image.resize(size=(self.depth_image_size, self.depth_image_size))
        pixel_values = torch.from_numpy(np.array(image)).float()
        pixel_values = pixel_values / 255
        pixel_values = 2 * pixel_values - 1
        pixel_values = rearrange(pixel_values, 'h w c -> c h w')
        return pixel_values

    def _process_depth_map(self, depth_img):
        """Process depth map image as target output."""
        if self.crop_image:
            depth_img = crop2square(depth_img)
        else:
            target_size = max(depth_img.size)
            depth_img = depth_img.resize(size=(target_size, target_size))
            
        depth_img = depth_img.resize(size=(self.depth_image_size, self.depth_image_size), resample=Image.BILINEAR)
        depth_array = np.array(depth_img).astype(np.float32)
        if len(depth_array.shape) == 2:
            pass
        elif len(depth_array.shape) == 3:
            depth_array = depth_array[:, :, 0] 
            
        d_min = depth_array.min()
        d_max = depth_array.max()
        if d_max - d_min > 1e-6:
            depth_normalized = (depth_array - d_min) / (d_max - d_min)
        else:
            depth_normalized = np.zeros_like(depth_array)
        
        depth_normalized = 2 * depth_normalized - 1
        depth_3ch = np.stack([depth_normalized, depth_normalized, depth_normalized], axis=0)
        depth_tensor = torch.from_numpy(depth_3ch).float()
        return depth_tensor

    def _process_depth_text(self):
        """Process text prompt for depth estimation."""
        depth_prompts = [
            "Generate the depth map of this image.",
            "Produce a depth estimation for this image.",
            "Create a depth map showing the distance of objects in this image.",
            "Estimate the depth information from this image.",
            "Generate a depth prediction for this image.",
        ]
        
        if random.uniform(0, 1) < self.depth_unconditional:
            final_prompt = "Generate an image."
        else:
            prompt = random.choice(depth_prompts)
            final_prompt = f"\n{prompt}"
        
        final_prompt = self.depth_prompt_template['INSTRUCTION'].format(input=final_prompt)
        input_ids = self.tokenizer.encode(final_prompt, add_special_tokens=True, return_tensors='pt')[0]
        
        # Insert IMAGE_TOKEN_INDEX placeholder for source image
        input_ids = torch.cat([
            input_ids[:3],
            torch.tensor([-200], dtype=torch.long),
            input_ids[3:],
        ], dim=0)
        
        return input_ids[:self.depth_max_length]

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, index):
        # --- 1. Get SFT Data ---
        sft_data = self.text_data[index]
        
        # Load Image (Shared)
        if sft_data.get('image', None) is not None:
            image_file = sft_data['image']
            rgb_path = os.path.join(self.image_folder, image_file)
            image = Image.open(rgb_path).convert('RGB')
            
            # Process for SFT
            if self.pad_image_to_square:
                sft_image = expand2square(
                    image,
                    tuple(int(x * 255) for x in self.image_processor.image_mean))
            else:
                sft_image = image
                
            sft_pixel_values = self.image_processor.preprocess(
                sft_image, return_tensors='pt')['pixel_values'][0]
            sft_data['pixel_values'] = sft_pixel_values
        else:
            # If no image, we can't do depth estimation for this sample
            # For Joint training, we require image. Raise Error.
            raise ValueError(f"Image missing for sample {index}, cannot perform joint training.")

        # --- 2. Get Depth Data ---
        # Try to find depth map
        depth_path = os.path.join(self.depth_folder, image_file)
        if not os.path.exists(depth_path):
            base, ext = os.path.splitext(depth_path)
            if os.path.exists(base + '.png'): depth_path = base + '.png'
            elif os.path.exists(base + '.jpg'): depth_path = base + '.jpg'
        
        if not os.path.exists(depth_path):
             raise FileNotFoundError(f"Depth map not found for {image_file}")

        try:
            depth_img = Image.open(depth_path)
            
            # Process for Depth Task
            # Input: Source RGB (same as SFT but potentially different transform)
            depth_source_pixel_values = self._process_depth_image(image)
            # Target: Depth Map
            depth_target_pixel_values = self._process_depth_map(depth_img)
            # Text: Prompt
            depth_input_ids = self._process_depth_text()
            
            # Combine into a Joint Data Dict
            # We use specific keys to distinguish SFT and Depth data
            combined_data = {
                'type': 'joint', # New type for Joint training
                
                # SFT Data
                'sft_input_ids': sft_data['input_ids'],
                'sft_labels': sft_data['labels'],
                'sft_attention_mask': sft_data.get('attention_mask'),
                'sft_pixel_values': sft_data['pixel_values'],
                
                # Depth Data
                'depth_source_pixel_values': depth_source_pixel_values,
                'depth_pixel_values': depth_target_pixel_values,
                'depth_input_ids': depth_input_ids,
                # depth_attention_mask is usually all 1s for short prompts, handled in collator or here?
                # We can add it here for safety
                'depth_attention_mask': torch.ones_like(depth_input_ids)
            }
            return combined_data
            
        except Exception as e:
            print(f"Error processing depth for {image_file}: {e}", flush=True)
            raise e

