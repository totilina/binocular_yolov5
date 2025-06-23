import torch
import torchvision.transforms as T
from PIL import Image
import os
from pathlib import Path

def visible_to_infrared(image_path, output_path):
    # 读取可见光图像并转换为Tensor
    transform = T.Compose([
        T.ToTensor(),
        T.Grayscale(num_output_channels=3)  # 保持3通道格式
    ])
    
    # 加载图像 (支持PNG/JPG等格式)
    img = Image.open(image_path).convert('RGB')
    tensor = transform(img)
    
    # 红外转换核心算法
    with torch.no_grad():
        # 增强红色通道并抑制蓝色通道
        infrared_tensor = torch.stack([
            tensor[0] * 0.8,   # 红色通道增强
            tensor[1] * 0.3,   # 绿色通道减弱
            tensor[2] * 0.1    # 蓝色通道大幅减弱
        ])
        
        # 添加热噪声模拟
        noise = torch.randn_like(infrared_tensor) * 0.05
        infrared_tensor = torch.clamp(infrared_tensor + noise, 0, 1)
        
        # 转换为单通道灰度
        infrared_tensor = infrared_tensor.mean(dim=0, keepdim=True)
    
    # 保存结果
    T.ToPILImage()(infrared_tensor).save(output_path)

def process_directory(input_dir, output_dir):
    """处理整个目录的可见光图片"""
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 获取所有图片文件
    img_exts = ('.jpg', '.jpeg', '.png')
    img_files = [f for f in os.listdir(input_dir) if f.lower().endswith(img_exts)]
    
    for img_file in img_files:
        src_path = os.path.join(input_dir, img_file)
        dest_path = os.path.join(output_dir, img_file)
        visible_to_infrared(src_path, dest_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, default = '../datasets/val/vis', help='可见光图片目录')
    parser.add_argument('--output-dir', type=str, default = '../datasets/val/ir1', help='红外图片输出目录')
    args = parser.parse_args()
    
    process_directory(args.input_dir, args.output_dir)