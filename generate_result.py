# c:\Users\bizhr\Desktop\a\pcu\binocular_yolov5\generate_result.py
import os
import glob
from pathlib import Path
import torch
from models.common import DetectMultiBackend
from utils.torch_utils import select_device, get_num_params, get_flops

# 加载模型计算参数量和计算量
device = select_device('0')
model = DetectMultiBackend('runs/train/binocular_exp/weights/best.pt', device=device)
imgsz = 640
params = get_num_params(model)
flops = get_flops(model, verbose=False, imgsz=640)


# 获取所有预测结果文件
pred_files = sorted(glob.glob('results/test_output15/labels/*.txt'))

# 创建结果文件
with open('result.txt', 'w') as f:
    # 写入模型参数量和计算量
    f.write(f"{params} {flops:.1f}\n")
    
    # 处理每个预测文件
    for pred_file in pred_files:
        img_name = Path(pred_file).stem + '.jpg'
        f.write(f"{img_name} ")
        
        # 读取预测结果
        if os.path.exists(pred_file) and os.path.getsize(pred_file) > 0:
            with open(pred_file, 'r') as pf:
                lines = pf.readlines()
                for line in lines:
                    # 解析预测结果 (class, x, y, w, h, conf)
                    parts = line.strip().split()
                    cls_id = int(parts[0])
                    x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    conf = float(parts[5]) if len(parts) > 5 else 1.0
                    
                    # 只保留置信度>=0.25的结果
                    if conf >= 0.25:
                        f.write(f"{x:g} {y:g} {w} {h} {conf.item()} {cls_id} ")
        
        f.write("\n")

print("结果文件已生成: result.txt")