import os
import sys
import torch
import warnings
import numpy as np
from PIL import Image
from einops import rearrange
from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE


class Harmon(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = False  # Harmon implementation in utils.py handles single image

    def check_install(self):
        """Check if Harmon dependencies are installed."""
        try:
            # Add Harmon repo root to sys.path (sibling folder named ``Harmon`` next to VLMEvalKit).
            harmon_path = os.path.join(os.path.dirname(__file__), '../../../Harmon')
            if os.path.exists(harmon_path) and harmon_path not in sys.path:
                sys.path.insert(0, harmon_path)
            
            import mmengine
            import xtuner
            from src.builder import BUILDER
        except Exception as e:
            logging.critical(
                'Please ensure Harmon and its dependencies (mmengine, xtuner) are available.')
            raise e

    def __init__(self, model_path=None, checkpoint_path=None, image_size=512, **kwargs):
        """
        Initialize Harmon model for VLM evaluation.
        
        Args:
            model_path: Path to model config file (Config)
            checkpoint_path: Path to model checkpoint (Checkpoint)
            image_size: Input image size (default: 512)
            **kwargs: Additional generation kwargs
        """
        self.check_install()
        
        from mmengine.config import Config
        from src.builder import BUILDER
        from xtuner.model.utils import guess_load_checkpoint
        
        assert model_path is not None, "model_path (config) must be provided"
        assert checkpoint_path is not None, "checkpoint_path must be provided"
        
        self.config_path = model_path
        self.checkpoint_path = checkpoint_path
        self.image_size = image_size
        
        # Load Config
        config = Config.fromfile(self.config_path)
        
        # Build Model
        print(f"Building Harmon model from {self.config_path}...")
        model = BUILDER.build(config.model).eval().cuda()
        model = model.to(model.dtype)
        
        # Load Checkpoint
        print(f"Loading checkpoint: {self.checkpoint_path}")
        if os.path.isdir(self.checkpoint_path):
            checkpoint = guess_load_checkpoint(self.checkpoint_path)
        else:
            checkpoint = torch.load(self.checkpoint_path, weights_only=False) # Harmon uses weights_only=False in utils.py
            
        info = model.load_state_dict(checkpoint, strict=False)
        
        # Filter out expected VAE missing keys
        unexpected_missing_keys = [k for k in info.missing_keys if not k.startswith('vae.')]
        if unexpected_missing_keys or info.unexpected_keys:
             print(f"Checkpoint loaded with unexpected issues: {info}")
        
        # Add special tokens
        special_tokens_dict = {'additional_special_tokens': ["<image>", ]}
        num_added_toks = model.tokenizer.add_special_tokens(special_tokens_dict)
        
        self.image_token_idx = model.tokenizer.encode("<image>", add_special_tokens=False)[-1]
        
        # Verify prompt template
        if not hasattr(model, 'prompt_template') or model.prompt_template is None:
            raise ValueError("Model does not have prompt_template attribute. Check model config.")
        if 'INSTRUCTION' not in model.prompt_template:
            raise ValueError("Model prompt_template does not have 'INSTRUCTION' key. Check model config.")
            
        self.model = model
        
        # Set default generation kwargs
        default_kwargs = dict(
            max_new_tokens=1024,
            do_sample=False,
            temperature=0.0, # Not used when do_sample=False
        )
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs
        
        warnings.warn(f'Harmon model loaded. Generation kwargs: {self.kwargs}')

    def expand2square(self, pil_img, background_color=(127, 127, 127)):
        """Expand image to square by padding"""
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    def process_image(self, image):
        """Process image for Harmon model"""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # Ensure image is RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = self.expand2square(image, (127, 127, 127))
        image = image.resize(size=(self.image_size, self.image_size))
        image = torch.from_numpy(np.array(image)).to(dtype=self.model.dtype, device=self.model.device)
        image = rearrange(image, 'h w c -> c h w')[None]
        image = 2 * (image / 255) - 1
        return image

    def generate_inner(self, message, dataset=None):
        """
        Generate response using Harmon model.
        """
        prompt, image_path = self.message_to_promptimg(message)
        
        if image_path is None:
            # Text-only input not fully supported by this specific pipeline logic, 
            # but we can try without image features if needed.
            # However, Harmon seems designed for VLM.
            return "Error: Image required for Harmon evaluation."

        # Process Image
        image_tensor = self.process_image(image_path)
        
        # Prepare Prompt
        prompt_template = self.model.prompt_template['INSTRUCTION']
        formatted_prompt = prompt_template.format(input="<image>\n" + prompt)
        
        # Replace <image> with multiple image tokens
        image_length = (self.image_size // 16) ** 2 + 64
        formatted_prompt = formatted_prompt.replace('<image>', '<image>' * image_length)
        
        # Tokenize
        input_ids = self.model.tokenizer.encode(
            formatted_prompt, add_special_tokens=True, return_tensors='pt').cuda()
            
        # Extract visual features
        with torch.no_grad():
            _, z_enc = self.model.extract_visual_feature(self.model.encode(image_tensor))
            
        # Create inputs embeddings
        inputs_embeds = z_enc.new_zeros(*input_ids.shape, self.model.llm.config.hidden_size)
        inputs_embeds[input_ids == self.image_token_idx] = z_enc.flatten(0, 1)
        
        # Fill in text embeddings
        mask_text = input_ids != self.image_token_idx
        inputs_embeds[mask_text] = self.model.llm.get_input_embeddings()(input_ids[mask_text])
        
        # Generate
        with torch.no_grad():
            output = self.model.llm.generate(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                do_sample=self.kwargs.get('do_sample', False),
                max_new_tokens=self.kwargs.get('max_new_tokens', 1024),
                eos_token_id=self.model.tokenizer.eos_token_id,
                pad_token_id=self.model.tokenizer.pad_token_id 
                if self.model.tokenizer.pad_token_id is not None else 
                self.model.tokenizer.eos_token_id,
                temperature=self.kwargs.get('temperature', 1.0) if self.kwargs.get('do_sample', False) else None
            )
            
        # Decode
        full_output = self.model.tokenizer.decode(output[0], skip_special_tokens=False)
        response = full_output.replace('<|im_end|>', '').replace('<|endoftext|>', '').strip()
        
        return response

    def use_custom_prompt(self, dataset):
        return False

