#!/usr/bin/env python
# coding=utf-8

'''
python scripts/parallel_geneval.py --gpus 3,4,5 --checkpoint work_dirs/exp18/iter_5000.pth --batch_size 4 --outdir dpg_exp18_5000 --mode dpgbench
python scripts/parallel_geneval.py --gpus 8,9 --checkpoint work_dirs/exp16/iter_3000.pth --batch_size 4 --outdir dpg_exp16_3000 --mode dpgbench --config /home/jixie/Harmon/configs/models/qwen2_5_0_5b_kl16_mar_b.py
'''

import os
import sys
import argparse
import multiprocessing as mp
import subprocess
import time
from itertools import cycle
import torch
import torch.distributed as dist

def run_geneval(gpu_id, args, start_idx, end_idx):
    cmd = [
        "CUDA_VISIBLE_DEVICES=" + str(gpu_id),
        "python", "scripts/geneval.py",
        f"--config={args.config}",
        f"--batch_size={args.batch_size}",
        f"--guidance_scale={args.guidance_scale}",
        f"--generation_timesteps={args.generation_timesteps}",
        f"--temperature={args.temperature}",
        f"--cfg_schedule={args.cfg_schedule}",
        f"--cfg_prompt='{args.cfg_prompt}'",
        f"--validation_prompts_file={args.validation_prompts_file}",
        f"--seed={args.seed}",
        f"--image_size={args.image_size}",
        f"--l={start_idx}",
        f"--r={end_idx}",
        f"--exp={args.exp}",
        f"--step={args.step}",
    ]
    
    # Optional CLI flags
    if args.outdir is not None:
        cmd.append(f"--outdir={args.outdir}")
    if args.checkpoint is not None:
        cmd.append(f"--checkpoint={args.checkpoint}")
    if args.use_template:
        cmd.append("--use_template")
    if args.remove_prefix:
        cmd.append("--remove_prefix")

    cmd_str = " ".join(cmd)
    print(f"GPU {gpu_id} running command: {cmd_str}")
    
    process = subprocess.Popen(
        cmd_str, 
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    prefix = f"[GPU {gpu_id}] "
    for line in process.stdout:
        # Drop blank lines and diffusion step tqdm lines (it/s) unless from our "Processing" tqdm.
        if line.strip() and ('it/s' not in line or 'Processing' in line):
            print(prefix + line.rstrip(), flush=True)
    
    return_code = process.wait()
    if return_code != 0:
        print(f"{prefix} exit: {return_code}", flush=True)
    else:
        print(f"{prefix} done.", flush=True)

def run_dpgbench(gpu_id, args, start_idx, end_idx):
    cmd = [
        "CUDA_VISIBLE_DEVICES=" + str(gpu_id),
        "python", "scripts/dpgbench.py",
        f"--config={args.config}",
        f"--batch_size=4",
        f"--guidance_scale={args.guidance_scale}",
        f"--generation_timesteps={args.generation_timesteps}",
        f"--temperature={args.temperature}",
        f"--cfg_schedule={args.cfg_schedule}",
        f"--cfg_prompt='{args.cfg_prompt}'",
        f"--seed={args.seed}",
        f"--image_size={args.image_size}",
        f"--l={start_idx}",
        f"--r={end_idx}",
        f"--prompts_file={args.prompts_file}",
    ]
    
    if args.outdir is not None:
        cmd.append(f"--outdir={args.outdir}")
    if args.checkpoint is not None:
        cmd.append(f"--checkpoint={args.checkpoint}")
    if args.use_template:
        cmd.append("--use_template")
    
    cmd_str = " ".join(cmd)
    print(f"GPU {gpu_id} running command: {cmd_str}")
    process = subprocess.Popen(
        cmd_str, 
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    prefix = f"[GPU {gpu_id}] "
    for line in process.stdout:
        # Drop blank lines and diffusion step tqdm lines (it/s) unless from our "Processing" tqdm.
        if line.strip() and ('it/s' not in line or 'Processing' in line):
            print(prefix + line.rstrip(), flush=True)
    
    return_code = process.wait()
    if return_code != 0:
        print(f"{prefix} exit: {return_code}", flush=True)
    else:
        print(f"{prefix} done.", flush=True)

def setup_distributed():
    """Initialize distributed training if needed"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        return True
    return False

def run_distributed_geneval(args, start_idx, end_idx):
    """Run geneval in distributed mode"""
    import json
    import random
    import numpy as np
    from src.builder import BUILDER
    from PIL import Image
    from mmengine.config import Config
    from tqdm import tqdm
    from xtuner.model.utils import guess_load_checkpoint
    
    def set_seed(seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{local_rank}"
    
    os.makedirs(args.outdir, exist_ok=True)
    config = Config.fromfile(args.config)
    model = BUILDER.build(config.model).eval().to(device)
    model = model.to(model.dtype)
    
    if args.checkpoint is not None:
        if rank == 0:
            print(f"Load checkpoint: {args.checkpoint}", flush=True)
        if os.path.isdir(args.checkpoint):
            checkpoint = guess_load_checkpoint(args.checkpoint)
        else:
            checkpoint = torch.load(args.checkpoint, weights_only=False, map_location=device)
    info = model.load_state_dict(checkpoint, strict=False)
    
    try:
        with open(args.validation_prompts_file) as fp:
            metadatas = [json.loads(line) for line in fp]
    except Exception as e:
        if rank == 0:
            print(f"Error loading validation prompts file: {e}")
        metadatas = [{"prompt": "a dog on the left and a cat on the right."}]
    
    for index, metadata in tqdm(enumerate(metadatas[start_idx:end_idx]), 
                                total=end_idx-start_idx,
                                desc=f"GPU {rank} processing"):
        set_seed(args.seed)
        actual_index = index + start_idx
        
        outpath = os.path.join(args.outdir, f"{actual_index:0>5}")
        os.makedirs(outpath, exist_ok=True)
        
        prompt = metadata.get("prompt", None)
        if rank == 0:
            print(f"Prompt ({actual_index: >3}/{len(metadatas)}): '{prompt}'")
        
        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)
        
        set_seed(args.seed)
        
        existing_images = [f for f in os.listdir(sample_path) if f.endswith('.png')]
        if len(existing_images) >= 12:
            continue
            
        full_prompt = f"Generate an image: {prompt}"
        class_info = model.prepare_text_conditions(full_prompt, args.cfg_prompt)
        
        input_ids = class_info['input_ids']
        attention_mask = class_info['attention_mask']
        
        assert len(input_ids) == 2
        
        batch_size = args.batch_size
        
        if args.guidance_scale != 1.0:
            input_ids = torch.cat([
                input_ids[0:1].expand(batch_size, -1),
                input_ids[1:2].expand(batch_size, -1),
            ])
            attention_mask = torch.cat([
                attention_mask[0:1].expand(batch_size, -1),
                attention_mask[1:2].expand(batch_size, -1),
            ])
        else:
            input_ids = input_ids[0:1].expand(batch_size, -1)
            attention_mask = attention_mask[0:1].expand(batch_size, -1)
        
        img_h = img_w = args.image_size // 16
        
        try:
            with torch.no_grad():
                samples = model.sample(input_ids=input_ids, 
                                      attention_mask=attention_mask,
                                      num_iter=args.generation_timesteps, 
                                      cfg=args.guidance_scale, 
                                      cfg_schedule=args.cfg_schedule,
                                      temperature=args.temperature, 
                                      progress=True, 
                                      image_shape=(img_h, img_w))
        except Exception as e:
            print(f"Error during sampling: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            continue
        for i in range(batch_size):
            sample = samples[i]
            sample = torch.clamp(127.5 * sample + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()
            sample = sample.transpose(1, 2, 0)
            
            out_path = os.path.join(sample_path, f"{i:05}.png")
            Image.fromarray(sample).save(out_path)
    
    if rank == 0:
        print("Done!")

def run_distributed_dpgbench(args, start_idx, end_idx):
    """Run dpgbench in distributed mode"""
    import json
    import random
    import numpy as np
    from src.builder import BUILDER
    from PIL import Image
    from mmengine.config import Config
    from tqdm import tqdm
    from xtuner.model.utils import guess_load_checkpoint
    
    def set_seed(seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{local_rank}"
    
    os.makedirs(args.outdir, exist_ok=True)
    config = Config.fromfile(args.config)
    model = BUILDER.build(config.model).eval().to(device)
    model = model.to(model.dtype)
    
    if args.checkpoint is not None:
        if rank == 0:
            print(f"Load checkpoint: {args.checkpoint}", flush=True)
        if os.path.isdir(args.checkpoint):
            checkpoint = guess_load_checkpoint(args.checkpoint)
        else:
            checkpoint = torch.load(args.checkpoint, weights_only=False, map_location=device)
    info = model.load_state_dict(checkpoint, strict=False)
    
    try:
        with open(args.prompts_file, 'r') as f:
            dataset = json.load(f)
    except Exception as e:
        if rank == 0:
            print(f"Error loading prompts file: {e}")
        dataset = {"default.txt": "a dog on the left and a cat on the right."}
    
    all_items = list(dataset.items())
    total_items = len(all_items)
    assigned_items = all_items[start_idx:end_idx]
    
    if rank == 0:
        print(f"Processing {len(assigned_items)} prompts (index {start_idx} to {end_idx-1} out of {total_items} total)")
    
    for idx, (key, prompt) in enumerate(tqdm(assigned_items, desc=f"GPU {rank} processing")):
        global_index = start_idx + idx
        
        print(f"Prompt ({global_index+1}/{total_items}, key={key}): '{prompt}'")
        
        batch_size = args.batch_size
        all_exist = True
        for img_idx in range(batch_size):
            out_path = os.path.join(args.outdir, f"{key.split('.')[-2]}_{img_idx}.jpg")
            if not os.path.exists(out_path):
                all_exist = False
                break
        
        if all_exist:
            print(f"Skipping generation for {key} - all images already exist")
            continue
            
        full_prompt = f"Generate an image: {prompt}"
        class_info = model.prepare_text_conditions(full_prompt, args.cfg_prompt)
        
        input_ids = class_info['input_ids']
        attention_mask = class_info['attention_mask']
        
        if args.guidance_scale != 1.0:
            input_ids = torch.cat([
                input_ids[0:1].expand(batch_size, -1),
                input_ids[1:2].expand(batch_size, -1),
            ])
            attention_mask = torch.cat([
                attention_mask[0:1].expand(batch_size, -1),
                attention_mask[1:2].expand(batch_size, -1),
            ])
        else:
            input_ids = input_ids[0:1].expand(batch_size, -1)
            attention_mask = attention_mask[0:1].expand(batch_size, -1)
        
        img_h = img_w = args.image_size // 16
        
        with torch.no_grad():
            samples = model.sample(input_ids=input_ids, 
                                  attention_mask=attention_mask,
                                  num_iter=args.generation_timesteps, 
                                  cfg=args.guidance_scale, 
                                  cfg_schedule=args.cfg_schedule,
                                  temperature=args.temperature, 
                                  progress=True, 
                                  image_shape=(img_h, img_w))
        
        for idx, sample in enumerate(samples):
            sample = torch.clamp(127.5 * sample + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()
            sample = sample.transpose(1, 2, 0)
            out_path = os.path.join(args.outdir, f"{key.split('.')[-2]}_{idx}.jpg")
            Image.fromarray(sample).save(out_path)
    
    if rank == 0:
        print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='config file path.', default='configs/models/qwen2_5_1_5b_kl16_mar_h.py')
    parser.add_argument("--checkpoint", "--ckpt", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--guidance_scale", "--cfg", type=float, default=3.0)
    parser.add_argument("--generation_timesteps", "--num_iter", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument('--cfg_schedule', type=str, default='constant')
    parser.add_argument('--cfg_prompt', type=str, default='Generate an image.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--validation_prompts_file', type=str,
                        default='prompts/geneval/evaluation_metadata.jsonl')
    parser.add_argument('--prompts_file', type=str,
                        default='prompts/dpgbench/prompts.json')
    parser.add_argument('--use_template', action='store_true')
    parser.add_argument('--exp', type=str, default='exp4')
    parser.add_argument('--step', type=int, default=0)
    parser.add_argument('--gpus', type=str, default='0,1,2,3,4,5,6,7')
    parser.add_argument('--total_prompts', type=int, default=None)
    parser.add_argument('--mode', type=str, default='geneval', choices=['geneval', 'dpgbench'])
    parser.add_argument('--remove_prefix', action='store_true')
    
    args = parser.parse_args()
    args.long = False
    
    # Check if running in distributed mode
    is_distributed = setup_distributed()
    
    if is_distributed:
        # Distributed mode: use torch.distributed
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        
        if rank == 0:
            print(f"Running in distributed mode with {world_size} processes")
        
        if args.outdir is None:
            if args.mode == 'geneval':
                args.outdir = args.exp + f"_{args.step}"
                if args.long:
                    args.outdir += "_long"
            else:
                args.outdir = f"dpg_harmon_results"
                if args.long:
                    args.outdir += "_long"
        
        if args.checkpoint is None and args.mode == 'geneval':
            args.checkpoint = f"work_dirs/{args.exp}/{args.exp}_{args.step}"
            if not os.path.exists(args.checkpoint):
                args.checkpoint = f"work_dirs/{args.exp}/iter_{args.step}.pth"

        os.makedirs(args.outdir, exist_ok=True)
        
        import json
        try:
            if args.mode == 'geneval':
                with open(args.validation_prompts_file) as fp:
                    prompts = [json.loads(line) for line in fp]
                total_prompts = args.total_prompts if args.total_prompts is not None else len(prompts)
                total_prompts = min(total_prompts, len(prompts))
            else:
                with open(args.prompts_file) as fp:
                    prompts = json.load(fp)
                # For dpgbench, prompts is a dict, convert to list of items for counting
                if isinstance(prompts, dict):
                    total_prompts = args.total_prompts if args.total_prompts is not None else len(prompts)
                    total_prompts = min(total_prompts, len(prompts))
                else:
                    total_prompts = args.total_prompts if args.total_prompts is not None else len(prompts)
                    total_prompts = min(total_prompts, len(prompts))
                if rank == 0:
                    print(f"Load {total_prompts} DPGBench prompts")
        except Exception as e:
            if rank == 0:
                print(f"Load prompts file error: {e}")
            prompts = {"default.txt": "a dog on the left and a cat on the right."}
            total_prompts = 1
        
        prompts_per_gpu = (total_prompts + world_size - 1) // world_size
        start_idx = rank * prompts_per_gpu
        end_idx = min(start_idx + prompts_per_gpu, total_prompts)
        
        if rank == 0:
            print(f"Totally {total_prompts} prompts will be divided into {world_size} tasks")
        print(f"GPU {rank} processing range: {start_idx} - {end_idx}")
        
        if args.mode == 'geneval':
            run_distributed_geneval(args, start_idx, end_idx)
        else:
            run_distributed_dpgbench(args, start_idx, end_idx)
        
        dist.barrier()
        if rank == 0:
            print("All processes done!")
    else:
        # Non-distributed mode: use multiprocessing
        gpu_ids = [int(gpu.strip()) for gpu in args.gpus.split(',')]
        num_gpus = len(gpu_ids)
        
        print(f"Use {num_gpus} GPUs: {gpu_ids}")
        
        if args.outdir is None:
            if args.mode == 'geneval':
                args.outdir = args.exp + f"_{args.step}"
                if args.long:
                    args.outdir += "_long"
            else:
                args.outdir = f"dpg_harmon_results"
                if args.long:
                    args.outdir += "_long"
        
        if args.checkpoint is None and args.mode == 'geneval':
            args.checkpoint = f"work_dirs/{args.exp}/{args.exp}_{args.step}"
            if not os.path.exists(args.checkpoint):
                args.checkpoint = f"work_dirs/{args.exp}/iter_{args.step}.pth"

        os.makedirs(args.outdir, exist_ok=True)
        
        import json
        try:
            if args.mode == 'geneval':
                with open(args.validation_prompts_file) as fp:
                    prompts = [json.loads(line) for line in fp]
                total_prompts = args.total_prompts if args.total_prompts is not None else len(prompts)
                total_prompts = min(total_prompts, len(prompts))
            else:
                with open(args.prompts_file) as fp:
                    prompts = json.load(fp)
                total_prompts = args.total_prompts if args.total_prompts is not None else len(prompts)
                total_prompts = min(total_prompts, len(prompts))
                print(f"Load {total_prompts} DPGBench prompts")
        except Exception as e:
            print(f"Load prompts file error: {e}")
            prompts = {"default.txt": "a dog on the left and a cat on the right."}
            total_prompts = 1
        
        prompts_per_gpu = (total_prompts + num_gpus - 1) // num_gpus
        ranges = []
        
        for i in range(num_gpus):
            start_idx = i * prompts_per_gpu
            end_idx = min((i + 1) * prompts_per_gpu, total_prompts)
            if start_idx < end_idx:
                ranges.append((start_idx, end_idx))

        print(f"Totally {total_prompts} prompts will be divided into {len(ranges)} tasks")

        processes = []
        for (start_idx, end_idx), gpu_id in zip(ranges, gpu_ids):
            print(f"GPU {gpu_id} processing range: {start_idx} - {end_idx}")
            target_func = run_geneval if args.mode == 'geneval' else run_dpgbench
            
            p = mp.Process(
                target=target_func,
                args=(gpu_id, args, start_idx, end_idx)
            )
            processes.append(p)
            p.start()
            time.sleep(1)
        
        for p in processes:
            p.join()
        
        print("All processes done!")
