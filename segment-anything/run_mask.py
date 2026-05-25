import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


def get_mask_edges(anns):
    if len(anns) == 0:
        return None
    h, w = anns[0]["segmentation"].shape
    edge_img = np.zeros((h, w), dtype=np.uint8)
    for ann in anns:
        mask = ann["segmentation"].astype(np.uint8) * 255
        edges = cv2.Canny(mask, 100, 200)
        edge_img = np.maximum(edge_img, edges)
    return edge_img


def generate_mask_edges(mask_generator, raw_image_bgr):
    img_rgb = cv2.cvtColor(raw_image_bgr, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(img_rgb)
    edge_img = get_mask_edges(masks)
    if edge_img is None:
        edge_img = np.zeros((raw_image_bgr.shape[0], raw_image_bgr.shape[1]), dtype=np.uint8)
    return edge_img


def process_file(mask_generator, filename, root_dir, outdir, pred_only, max_size=None, log_fn=print):
    if root_dir is not None:
        rel_path = os.path.relpath(os.path.abspath(filename), root_dir)
        rel_dir = os.path.dirname(rel_path)
        base_name = os.path.splitext(os.path.basename(rel_path))[0] + ".png"
        target_dir = os.path.join(outdir, rel_dir)
        os.makedirs(target_dir, exist_ok=True)
        out_path = os.path.join(target_dir, base_name)
    else:
        out_path = os.path.join(
            outdir, os.path.splitext(os.path.basename(filename))[0] + ".png"
        )

    if os.path.exists(out_path):
        log_fn(f"  -> Skip (exists): {out_path}")
        return

    raw_image = cv2.imread(filename)
    if raw_image is None:
        log_fn(f"  -> Skip (unreadable): {filename}")
        return

    orig_h, orig_w = raw_image.shape[:2]
    process_image = raw_image
    if max_size is not None and max(orig_h, orig_w) > max_size:
        scale = max_size / max(orig_h, orig_w)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        process_image = cv2.resize(raw_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        log_fn(f"  -> Resize {orig_w}x{orig_h} -> {new_w}x{new_h}")

    try:
        edge_img = generate_mask_edges(mask_generator, process_image)
    except RuntimeError as e:
        if "INT_MAX" in str(e) or "nonzero" in str(e):
            log_fn(f"  -> Skip (too large): {filename} ({orig_w}x{orig_h})")
            return
        log_fn(f"  -> Error {filename}: {e}")
        return
    except Exception as e:
        log_fn(f"  -> Error {filename}: {e}")
        return

    if process_image is not raw_image:
        edge_img = cv2.resize(edge_img, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    edge_vis = cv2.cvtColor(edge_img, cv2.COLOR_GRAY2BGR) if edge_img.ndim == 2 else edge_img

    if pred_only:
        cv2.imwrite(out_path, edge_vis)
    else:
        if edge_vis.shape[0] != raw_image.shape[0]:
            new_w = int(edge_vis.shape[1] * raw_image.shape[0] / edge_vis.shape[0])
            edge_vis = cv2.resize(edge_vis, (new_w, raw_image.shape[0]))
        split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
        combined = cv2.hconcat([raw_image, split_region, edge_vis])
        cv2.imwrite(out_path, combined)


def main():
    p = argparse.ArgumentParser(description="SAM automatic masks")
    p.add_argument("--img-path", required=True, help="image file, directory (recursive), or .txt list")
    p.add_argument("--outdir", default="./vis_mask", help="output root")
    p.add_argument("--checkpoint", default="sam_vit_h_4b8939.pth", help="SAM .pth path")
    p.add_argument("--model-type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    p.add_argument("--pred-only", action="store_true", help="save edge only (not side-by-side)")
    p.add_argument("--max-size", type=int, default=2048, help="max long edge; 0 = no resize")
    p.add_argument("--device", default=None, help='e.g. "cuda:0" or "cpu"; default: cuda:0 if available else cpu')
    args = p.parse_args()

    if args.device is not None:
        device = args.device
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    max_size = args.max_size if args.max_size > 0 else None

    print(f"Device: {device}")
    if max_size:
        print(f"Max size: {max_size}")

    root_dir = None
    if os.path.isfile(args.img_path):
        if args.img_path.endswith(".txt"):
            with open(args.img_path) as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        root_dir = os.path.abspath(args.img_path)
        filenames = glob.glob(os.path.join(args.img_path, "**/*"), recursive=True)

    filenames = sorted(f for f in filenames if os.path.isfile(f))
    print(f"Files: {len(filenames)}")
    os.makedirs(args.outdir, exist_ok=True)

    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    for path in tqdm(filenames, desc="Edges", ncols=80):
        process_file(
            mask_generator,
            path,
            root_dir,
            args.outdir,
            args.pred_only,
            max_size=max_size,
            log_fn=tqdm.write,
        )

    print("Done.")


if __name__ == "__main__":
    main()
