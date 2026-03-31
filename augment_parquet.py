"""
运行命令示例:
python augment_parquet.py \
    --input_dir /mnt/hwdata/wangsen/hw_data/Converted/test/data/chunk-000 \
    --ckpt /mnt/hwdata/wangsen/gensplat_ckpt \
    --trans_noise 0.02
"""
    
import os
import io
import glob
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

import sys
# 确保能够导入 src 下的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model.model.gensplat import GenSplat

def setup_args():
    parser = argparse.ArgumentParser(description='Parquet Data Augmentation via GenSplat NVS (Single-View)')
    parser.add_argument('--input_dir', type=str, required=True, 
                        help='输入数据目录，例如: /mnt/hwdata/.../chunk-000')
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='输出数据目录，默认在 input_dir 同级自动加 -new')
    parser.add_argument('--ckpt', type=str, default="/mnt/hwdata/wangsen/gensplat_ckpt", 
                        help='模型权重文件夹路径或 HuggingFace 模型名称')
    parser.add_argument('--trans_noise', type=float, default=0.02, 
                        help='相机平移扰动强度 (米)，默认 2cm')
    return parser.parse_args()

def tensor_to_bytes(tensor_img):
    """
    将 [3, H, W] 且范围在 [0, 1] 的 Tensor 转换回 PNG bytes
    """
    img_np = (tensor_img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    
    buf = io.BytesIO()
    img_pil.save(buf, format='PNG')
    return buf.getvalue()

def perturb_extrinsics(extrinsics, noise_std=0.02):
    """
    对相机外参(Extrinsics)进行微小扰动，实现新视角数据增强
    extrinsics shape: [B, K, 4, 4]
    """
    new_extrinsics = extrinsics.clone()
    noise = torch.randn_like(new_extrinsics[:, :, :3, 3]) * noise_std
    new_extrinsics[:, :, :3, 3] += noise
    return new_extrinsics

def main():
    args = setup_args()
    
    # 1. 设置输入输出路径
    input_dir = args.input_dir.rstrip('/')
    if args.output_dir is None:
        output_dir = input_dir + "-new"
    else:
        output_dir = args.output_dir
        
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 输入路径: {input_dir}")
    print(f"📁 输出路径: {output_dir}")

    # 2. 加载模型
    print(f"🚀 正在加载模型: {args.ckpt}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GenSplat.from_pretrained(args.ckpt)
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    parquet_files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    if len(parquet_files) == 0:
        print("❌ 没有找到任何 .parquet 文件！")
        return

    recon_cam_columns = ['observation.image1', 'observation.image2'] # 独立进行单视角重建
    keep_cam_columns = ['observation.image3'] # 不参与重建
    all_cam_columns = recon_cam_columns + keep_cam_columns
    
    to_tensor = T.ToTensor()
    PATCH_SIZE = 14 # 适配模型要求的高宽倍数

    # 3. 开始遍历处理
    for file_path in parquet_files:
        file_name = os.path.basename(file_path)
        out_file_path = os.path.join(output_dir, file_name)
        
        print(f"\n📄 正在处理: {file_name}")
        df = pd.read_parquet(file_path)
        
        new_cam_data = {col: [] for col in all_cam_columns}
        
        for index, row in tqdm(df.iterrows(), total=len(df), desc=f"处理 {file_name}"):
            
            # --- A. 对 cam1 和 cam2 进行【独立】单视角处理 ---
            for col in recon_cam_columns:
                img_bytes = row[col]['bytes']
                img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                
                # 裁剪以适配 PATCH_SIZE
                orig_w, orig_h = img.size
                new_h = (orig_h // PATCH_SIZE) * PATCH_SIZE
                new_w = (orig_w // PATCH_SIZE) * PATCH_SIZE
                img_cropped = F.center_crop(img, (new_h, new_w))
                
                img_tensor = to_tensor(img_cropped) # [3, H, W]
                
                # 构建单视角 Batch: [Batch=1, K=1, C=3, H, W]
                images_t = img_tensor.unsqueeze(0).unsqueeze(0).to(device)
                b, v, c, h, w = images_t.shape # 此时 v (视角数) = 1
                
                with torch.no_grad():
                    # 单图送入模型推理
                    gaussians, pred_context_pose = model.inference(images_t)
                    
                    # 扰动单图的外参
                    novel_extrinsics = perturb_extrinsics(pred_context_pose['extrinsic'], args.trans_noise)
                    
                    # 渲染单图的新视角
                    near = torch.ones(1, v, device=device) * 0.01
                    far = torch.ones(1, v, device=device) * 100
                    
                    output = model.decoder.forward(
                        gaussians, 
                        novel_extrinsics, 
                        pred_context_pose['intrinsic'].float(), 
                        near, 
                        far, 
                        (h, w)
                    )
                    
                    # 取出渲染好的单张图片 [3, H, W]
                    rendered_image = output.color[0][0]
                
                # 保存回对应列的数据结构
                new_bytes = tensor_to_bytes(rendered_image)
                new_dict = row[col].copy() if isinstance(row[col], dict) else {}
                new_dict['bytes'] = new_bytes
                new_cam_data[col].append(new_dict)
                
            # --- B. 对 cam3 保持原样 ---
            for col in keep_cam_columns:
                new_cam_data[col].append(row[col].copy() if isinstance(row[col], dict) else {})

        # 4. 替换数据并保存
        for col in all_cam_columns:
            df[col] = new_cam_data[col]
            
        df.to_parquet(out_file_path)
        print(f"✅ 保存完毕: {out_file_path}")

if __name__ == "__main__":
    main()