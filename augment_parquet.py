"""
Example:

python augment_parquet.py \
    --input_dir /path/to/chunk-000 \
    --ckpt /path/to/gensplat_hf_export \
    --trans_noise 0.02
"""

import argparse
import glob
import io
import os
import sys

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model.model.gensplat import GenSplat


def setup_args():
    parser = argparse.ArgumentParser(
        description="Parquet data augmentation with single-view GenSplat novel-view synthesis."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input parquet files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for augmented parquet files. Defaults to <input_dir>-new.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/mnt/hwdata/wangsen/gensplat_ckpt",
        help="Model path or Hugging Face identifier accepted by GenSplat.from_pretrained.",
    )
    parser.add_argument(
        "--trans_noise",
        type=float,
        default=0.02,
        help="Translation noise applied to the predicted camera pose in scene units.",
    )
    return parser.parse_args()


def tensor_to_bytes(tensor_img):
    """Convert a [3, H, W] tensor in [0, 1] to PNG bytes."""
    img_np = (
        tensor_img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
    ).astype(np.uint8)
    img_pil = Image.fromarray(img_np)

    buffer = io.BytesIO()
    img_pil.save(buffer, format="PNG")
    return buffer.getvalue()


def perturb_extrinsics(extrinsics, noise_std=0.02):
    """Apply a small random translation to camera extrinsics."""
    new_extrinsics = extrinsics.clone()
    noise = torch.randn_like(new_extrinsics[:, :, :3, 3]) * noise_std
    new_extrinsics[:, :, :3, 3] += noise
    return new_extrinsics


def main():
    args = setup_args()

    input_dir = args.input_dir.rstrip("/\\")
    output_dir = args.output_dir or f"{input_dir}-new"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    print(f"Loading model from: {args.ckpt}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GenSplat.from_pretrained(args.ckpt)
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    parquet_files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    if not parquet_files:
        print("No parquet files were found.")
        return

    recon_cam_columns = ["observation.image1", "observation.image2"]
    keep_cam_columns = ["observation.image3"]
    all_cam_columns = recon_cam_columns + keep_cam_columns

    to_tensor = T.ToTensor()
    patch_size = 14

    for file_path in parquet_files:
        file_name = os.path.basename(file_path)
        out_file_path = os.path.join(output_dir, file_name)

        print(f"\nProcessing: {file_name}")
        df = pd.read_parquet(file_path)

        new_cam_data = {col: [] for col in all_cam_columns}

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Augmenting {file_name}"):
            for col in recon_cam_columns:
                img_bytes = row[col]["bytes"]
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

                orig_w, orig_h = img.size
                new_h = (orig_h // patch_size) * patch_size
                new_w = (orig_w // patch_size) * patch_size
                img_cropped = F.center_crop(img, (new_h, new_w))

                img_tensor = to_tensor(img_cropped)
                images_t = img_tensor.unsqueeze(0).unsqueeze(0).to(device)
                _, view_count, _, h, w = images_t.shape

                with torch.no_grad():
                    gaussians, pred_context_pose = model.inference(images_t)
                    novel_extrinsics = perturb_extrinsics(
                        pred_context_pose["extrinsic"],
                        args.trans_noise,
                    )

                    near = torch.ones(1, view_count, device=device) * 0.01
                    far = torch.ones(1, view_count, device=device) * 100

                    output = model.decoder.forward(
                        gaussians,
                        novel_extrinsics,
                        pred_context_pose["intrinsic"].float(),
                        near,
                        far,
                        (h, w),
                    )
                    rendered_image = output.color[0][0]

                new_bytes = tensor_to_bytes(rendered_image)
                new_dict = row[col].copy() if isinstance(row[col], dict) else {}
                new_dict["bytes"] = new_bytes
                new_cam_data[col].append(new_dict)

            for col in keep_cam_columns:
                new_cam_data[col].append(
                    row[col].copy() if isinstance(row[col], dict) else {}
                )

        for col in all_cam_columns:
            df[col] = new_cam_data[col]

        df.to_parquet(out_file_path)
        print(f"Saved: {out_file_path}")


if __name__ == "__main__":
    main()
