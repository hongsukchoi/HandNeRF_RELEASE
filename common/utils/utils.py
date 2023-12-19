import os
import sys
import gzip
import json
import torch

from config import cfg

def read_jgz(file_path: str):
    with gzip.open(file_path, 'r') as f:
        json_bytes = f.read()
    json_str = json_bytes.decode('utf-8')          
    data = json.loads(json_str)  
    return data 


def make_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def increase_object_ray_ratio_grad(global_iter, steps=4000):
    cfg_changed = False
    if (global_iter > 0) and (global_iter % steps == 0) and (cfg.object_sample_ratio < cfg.final_object_ratio):
        cfg.object_sample_ratio = cfg.object_sample_ratio + 0.1
        cfg_changed = True
    return cfg_changed
