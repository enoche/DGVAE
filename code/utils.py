import csv
import json
import os

import torch


def get_device():
    return torch.device(f'cuda:{os.environ["CUDA_DEVICE"]}' if torch.cuda.is_available() else 'cpu')


def write_dict_to_csv(data_dict, out_file):
    with open(out_file, 'w', encoding='utf-8') as f:
        w = csv.DictWriter(f, data_dict.keys())
        w.writeheader()
        w.writerow(data_dict)


def save_to_json(output_dict, out_file):
    with open(out_file, 'w', encoding='utf-8', errors='surrogatepass') as writer:
        json.dump(output_dict, writer, indent=4, ensure_ascii=False)


def load_json(inp_file):
    return json.load(open(inp_file, 'r', encoding='utf-8'))