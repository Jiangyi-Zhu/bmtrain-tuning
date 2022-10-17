import sys
import time
from datetime import datetime, timedelta, timezone
import argparse

def get_current_time():
    utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)
    SHA_TZ = timezone(timedelta(hours=8),name='Asia/Shanghai',)
    beijing_now = utc_now.astimezone(SHA_TZ)
    current_time = str(beijing_now)[:10]+'_'+str(beijing_now)[11:13]+'-'+str(beijing_now)[14:16]
    return current_time

class Logger(object):
    def __init__(self, path="Default.log"):
        self.terminal = sys.stdout
        file_path = f'{path}/print_log.txt'
        self.log = open(file_path, "a",encoding='utf-8')     #文件权限为'a'，追加模式
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", type=str)
    parser.add_argument("--source", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--model_class", type=str)
    parser.add_argument("--language", type=str)
    parser.add_argument("--soft_layer", type=int)
    parser.add_argument("--adapter_bottleneck_dim", type=int)
    parser.add_argument("--adapter_init_std", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dtype", type=str)
    parser.add_argument("--lr_scheduler", type=str)
    parser.add_argument("--soft_template_load_path", type=str)
    parser.add_argument("--adapter_load_path", type=str)
    parser.add_argument("--checkpoint_save_path", type=str)
    return parser.parse_args()