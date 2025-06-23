import yaml
import subprocess
from pathlib import Path

def load_config(yaml_path):
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    return config

def build_command(config):
    base_cmd = [
        'python', 'train.py',
        f'--weights', str(Path(config['weights']).resolve()),
        f'--cfg', config['cfg'],
        f'--data', config['data'],
        f'--epochs', str(config['epochs']),
        f'--batch-size', str(config['batch_size']),
        f'--imgsz', str(config['imgsz']),
        f'--device', config['device'],
        f'--workers', str(config['workers']),
        f'--project', config['project'],
        f'--name', config['name'],
        f'--seed', str(config['seed'])
    ]
    
    # 添加布尔型参数
    if config.get('cos_lr'):
        base_cmd.append('--cos-lr')
    if config.get('label_smoothing'):
        base_cmd.extend(['--label-smoothing', str(config['label_smoothing'])])
    
    # 生成hyp.yaml临时文件
    hyp_path = Path('temp_hyp.yaml')
    with open(hyp_path, 'w') as f:
        yaml.dump(config['hyp'], f)
    base_cmd.extend(['--hyp', str(hyp_path.resolve())])
    
    return base_cmd

if __name__ == '__main__':
    config = load_config('runs/o.yaml')
    cmd = build_command(config)
    print('运行命令:\n' + ' '.join(cmd))
    subprocess.run(cmd)