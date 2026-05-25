#!/usr/bin/env python3
"""Pixel / depth / segmentation reconstruction from one image; single checkpoint."""
import argparse
import os
import sys

import numpy as np
import torch
from einops import rearrange
from mmengine.config import Config
from PIL import Image

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.builder import BUILDER
from xtuner.model.utils import guess_load_checkpoint

MODE_PROMPT = {
    "pixel": "Describe the image in detail.",
    "depth": "Generate the depth map of this image.",
    "segmentation": "Generate the segmentation mask of this image.",
}


def load_model(config_path: str, checkpoint: str):
    cfg = Config.fromfile(config_path)
    if "pretrained_pth" in cfg.model:
        cfg.model.pretrained_pth = None
    model = BUILDER.build(cfg.model).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    model = model.to(model.dtype)
    if os.path.isdir(checkpoint):
        state = guess_load_checkpoint(checkpoint)
    else:
        try:
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=False)
    return model


def main():
    default_img = os.path.join(_ROOT, "assets", "image.jpg")
    p = argparse.ArgumentParser()
    p.add_argument("--input_image", default=default_img)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--mode", choices=list(MODE_PROMPT), required=True)
    p.add_argument("--config", default="configs/examples/UniMRG.py")
    p.add_argument("--output", default=None)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--num_iter", type=int, default=32)
    p.add_argument("--cfg", type=float, default=3.0)
    args = p.parse_args()

    model = load_model(args.config, args.checkpoint)
    pil = Image.open(args.input_image).convert("RGB").resize((args.image_size, args.image_size))
    x = torch.from_numpy(np.array(pil)).to(device=model.device, dtype=model.dtype)
    x = rearrange(x, "h w c -> 1 c h w") / 255.0 * 2.0 - 1.0

    llm = model.llm_model
    old_cache, llm.config.use_cache = llm.config.use_cache, True
    was_gc = getattr(llm, "is_gradient_checkpointing", False)
    if was_gc:
        model.gradient_checkpointing_disable()
    try:
        with torch.no_grad():
            pred = model.sample_recon(
                x, prompt=MODE_PROMPT[args.mode], num_iter=args.num_iter, cfg=args.cfg, progress=True
            )[0]
    finally:
        llm.config.use_cache = old_cache
        if was_gc:
            model.gradient_checkpointing_enable()

    out = pred.cpu().float()
    out = ((out + 1) / 2).clamp(0, 1).numpy().transpose(1, 2, 0)
    out = (out * 255).astype(np.uint8)
    outp = args.output or os.path.join(
        os.path.dirname(os.path.abspath(args.input_image)) or ".",
        f"{os.path.splitext(os.path.basename(args.input_image))[0]}_{args.mode}.png",
    )
    Image.fromarray(out).save(outp)
    print(outp)


if __name__ == "__main__":
    main()
