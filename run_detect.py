import subprocess
from pathlib import Path

# 基础配置
ROOT = Path(__file__).parent.absolute()  # 脚本所在目录
python_cmd = "python"  # 如果使用虚拟环境可改为 "path/to/python"

# 可修改参数配置
config = {
    "weights": ROOT / "runs/train/e/weights/best.pt",
    "source": ROOT / "../val/vis",  # 输入源路径
    "conf_thres": 0.3,               # 置信度阈值
    "iou_thres": 0.35,               # IOU阈值
    "imgsz": [680],             # 推理尺寸 [height, width]
    "device": "0",                   # 使用设备
    "view_img": False,               # 是否显示结果
    "save_txt": True,                # 是否保存txt结果
    "save_conf": False,               # 是否保存置信度
    # "classes": None,                 # 过滤特定类别
    "nosave": True,
    "augment": True,
    "project": "results",
    "name":"val_output"
    
}

# 构建命令参数
cmd = [
    python_cmd,
    str(ROOT / "detect.py"),
    f"--weights {str(config['weights'])}",
    f"--source {str(config['source'])}",
    f"--conf-thres {config['conf_thres']}",
    f"--iou-thres {config['iou_thres']}",
    f"--device {config['device']}",
    f"--imgsz {' '.join(map(str, config['imgsz']))}",
    f"--project {str(config['project'])}",
    f"--name {config['name']}",
    # f"--line-thickness {config['line_thickness']}",
    # f"--vid-stride {config['vid_stride']}"
]

# 添加布尔型参数
if config["view_img"]:
    cmd.append("--view-img")
if config["save_txt"]:
    cmd.append("--save-txt")
if config["save_conf"]:
    cmd.append("--save-conf")
if config["augment"]:
    cmd.append("--augment")
if config["nosave"]:
    cmd.append("--nosave")

# # 添加类别过滤
# if config["classes"]:
#     cmd.append(f"--classes {' '.join(map(str, config['classes']))}")

# 运行命令
subprocess.run(" ".join(cmd), shell=True)