import os
import glob
import json
import torch
import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm
import argparse

CHECKPOINT_DIR = "/mnt/sda1/yxz/Flame-moe/converted_checkpoints"

def get_checkpoints():
    subdirs = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "iter_*")))
    checkpoints = []
    for d in subdirs:
        try:
            iter_num = int(os.path.basename(d).split("_")[1])
            checkpoints.append((iter_num, d))
        except ValueError:
            continue
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints

def extract_all_parameters(output_base_dir):
    checkpoints = get_checkpoints()
    if not checkpoints:
        print("No checkpoints found.")
        return

    first_ckpt_path = checkpoints[0][1]
    print(f"Using {first_ckpt_path} as reference for index files.")

    # Load Model Index
    model_index_path = os.path.join(first_ckpt_path, "model.safetensors.index.json")
    if not os.path.exists(model_index_path):
        print(f"Model index not found at {model_index_path}")
        return
    with open(model_index_path, 'r') as f:
        model_idx = json.load(f)
    model_weight_map = model_idx.get("weight_map", {})

    # Load Optimizer Index
    opt_index_path = os.path.join(first_ckpt_path, "optimizer.safetensors.index.json")
    if not os.path.exists(opt_index_path):
        print(f"Optimizer index not found at {opt_index_path}")
        return
    with open(opt_index_path, 'r') as f:
        opt_idx = json.load(f)
    opt_weight_map = opt_idx.get("weight_map", {})

    all_params = list(model_weight_map.keys())
    print(f"Found {len(all_params)} parameters to process.")

    for param_name in tqdm(all_params, desc="Processing Parameters"):
        # Create output directory for this parameter
        safe_param_name = param_name.replace("/", "_")
        param_output_dir = os.path.join(output_base_dir, safe_param_name)
        os.makedirs(param_output_dir, exist_ok=True)

        # Determine the correct chained prefix for optimizer states
        # experts use chained_1, everything else uses chained_0
        chained_prefix = "chained_0"
        test_key = f"chained_1.optimizer.state.exp_avg.{param_name}"
        if test_key in opt_weight_map:
            chained_prefix = "chained_1"

        # Define the tasks (tensor types to extract)
        tasks = {
            "weight": {
                "key": param_name,
                "map": model_weight_map,
                "source": "model"
            },
            "momentum": {
                "key": f"{chained_prefix}.optimizer.state.exp_avg.{param_name}",
                "map": opt_weight_map,
                "source": "optimizer"
            },
            "variance": {
                "key": f"{chained_prefix}.optimizer.state.exp_avg_sq.{param_name}",
                "map": opt_weight_map,
                "source": "optimizer"
            },
            "master_weight": {
                "key": f"{chained_prefix}.optimizer.state.fp32_param.{param_name}",
                "map": opt_weight_map,
                "source": "optimizer"
            }
        }

        # For each type (weight, momentum, etc.), collect data across checkpoints
        for task_type, task_info in tasks.items():
            tensor_key = task_info["key"]
            weight_map = task_info["map"]
            
            output_file = os.path.join(param_output_dir, f"{task_type}.safetensors")
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                # print(f"Skipping {task_type} for {param_name}, already exists.")
                continue

            if tensor_key not in weight_map:
                # print(f"Warning: {tensor_key} not found in index.")
                continue

            target_filename = weight_map[tensor_key]
            tensors_to_save = {}
            found_count = 0

            # Iterate checkpoints for this specific tensor type
            for iter_num, ckpt_path in checkpoints:
                file_path = os.path.join(ckpt_path, target_filename)
                
                # Verify file exists
                if not os.path.exists(file_path):
                    continue

                try:
                    with safe_open(file_path, framework="pt", device="cpu") as f:
                        if tensor_key in f.keys():
                            tensor = f.get_tensor(tensor_key)
                            tensors_to_save[f"step_{iter_num}"] = tensor
                            found_count += 1
                except Exception as e:
                    print(f"Error reading {file_path} for {tensor_key}: {e}")

            if found_count > 0:
                try:
                    save_file(tensors_to_save, output_file)
                except Exception as e:
                    print(f"Failed to save {output_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Extract all tensors from checkpoints into containers.")
    parser.add_argument("--output_dir", default="extracted_containers_all", help="Directory to save the containers")
    args = parser.parse_args()

    extract_all_parameters(args.output_dir)

if __name__ == "__main__":
    main()
