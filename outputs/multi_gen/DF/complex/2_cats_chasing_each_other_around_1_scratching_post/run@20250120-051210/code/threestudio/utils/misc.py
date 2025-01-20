import gc
import os
import re

import tinycudann as tcnn
import torch
from packaging import version
import importlib

from threestudio.utils.config import config_to_primitive
from threestudio.utils.typing import *

import matplotlib.pyplot as plt
import numpy as np
import cv2

def create_subplot(fig, position, x, y, z, elev, azim, title):
    ax = fig.add_subplot(position, projection='3d')
    ax.scatter(x, y, z)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    return ax

def plot_bool_tensor(tensor, filepath):
    coords = torch.nonzero(tensor)
    x, y, z = coords[:, 1].cpu(), coords[:, 2].cpu(), coords[:, 3].cpu()
    
    fig = plt.figure(figsize=(18, 6))
    create_subplot(fig, 131, x.numpy(), y.numpy(), z.numpy(), 90, -90, 'Top View')
    create_subplot(fig, 132, x.numpy(), y.numpy(), z.numpy(), 0, -90, 'Front View')
    create_subplot(fig, 133, x.numpy(), y.numpy(), z.numpy(), 0, 0, 'Right View')
    plt.savefig(filepath)


def plot_bool_tensor_new(tensor, filepath, downsize=1):
    coords = torch.nonzero(tensor)
    x, y, z = coords[:, 1].cpu().numpy(), coords[:, 2].cpu().numpy(), coords[:, 3].cpu().numpy()

    grid_size = downsize

    # Downsampling: Grid-based sampling
    # Compute the indices of the representative points
    x_grid = (x // grid_size) * grid_size
    y_grid = (y // grid_size) * grid_size
    z_grid = (z // grid_size) * grid_size

    unique_indices = np.unique(np.vstack((x_grid, y_grid, z_grid)), axis=1)
    fig = plt.figure(figsize=(18, 6))
    create_subplot(fig, 131, *unique_indices, 90, -90, 'Top View')
    create_subplot(fig, 132, *unique_indices, 0, -90, 'Front View')
    create_subplot(fig, 133, *unique_indices, 0, 0, 'Right View')
    plt.savefig(filepath)


def parse_version(ver: str):
    return version.parse(ver)


def get_rank():
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


def get_device():
    return torch.device(f"cuda:{get_rank()}")


def load_module_weights(
    path, module_name=None, ignore_modules=None, map_location=None
) -> Tuple[dict, int, int]:
    if module_name is not None and ignore_modules is not None:
        raise ValueError("module_name and ignore_modules cannot be both set")
    if map_location is None:
        map_location = get_device()

    ckpt = torch.load(path, map_location=map_location)
    state_dict = ckpt["state_dict"]
    state_dict_to_load = state_dict

    if ignore_modules is not None:
        state_dict_to_load = {}
        for k, v in state_dict.items():
            ignore = any(
                [k.startswith(ignore_module + ".") for ignore_module in ignore_modules]
            )
            if ignore:
                continue
            state_dict_to_load[k] = v

    if module_name is not None:
        state_dict_to_load = {}
        for k, v in state_dict.items():
            m = re.match(rf"^{module_name}\.(.*)$", k)
            if m is None:
                continue
            state_dict_to_load[m.group(1)] = v

    return state_dict_to_load, ckpt["epoch"], ckpt["global_step"]


def C(value: Any, epoch: int, global_step: int) -> float:
    if isinstance(value, int) or isinstance(value, float):
        pass
    else:
        value = config_to_primitive(value)
        if not isinstance(value, list):
            raise TypeError("Scalar specification only supports list, got", type(value))
        if len(value) == 3:
            value = [0] + value
        assert len(value) == 4
        start_step, start_value, end_value, end_step = value
        if isinstance(end_step, int):
            current_step = global_step
            value = start_value + (end_value - start_value) * max(
                min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0
            )
        elif isinstance(end_step, float):
            current_step = epoch
            value = start_value + (end_value - start_value) * max(
                min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0
            )
    return value


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    tcnn.free_temporary_memory()


def finish_with_cleanup(func: Callable):
    def wrapper(*args, **kwargs):
        out = func(*args, **kwargs)
        cleanup()
        return out

    return wrapper


def _distributed_available():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def barrier():
    if not _distributed_available():
        return
    else:
        torch.distributed.barrier()


def broadcast(tensor, src=0):
    if not _distributed_available():
        return tensor
    else:
        torch.distributed.broadcast(tensor, src=src)
        return tensor


def enable_gradient(model, enabled: bool = True) -> None:
    for param in model.parameters():
        param.requires_grad_(enabled)

def find_center_of_box(xmin, ymin, zmin, xmax, ymax, zmax, device=None):
    """Find the center of a 3D box given its min and max coordinates."""
    if device is not None:
        return torch.tensor([(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2], device=device)
    else:
        return torch.tensor([(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2])
    
def find_width_of_box(xmin, ymin, zmin, xmax, ymax, zmax, device=None):
    """Find the center of a 3D box given its min and max coordinates."""
    if device is not None:
        return torch.tensor([xmax - xmin, ymax - ymin, zmax - zmin], device=device)
    else:
        return torch.tensor([xmax - xmin, ymax - ymin, zmax - zmin])



def process_gpt_file_zero123(file_path, target_range = [-2, 2], fixed_mapping = True, return_full_prompt = False):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    original_data = []
    for i, line in enumerate(lines):
        # match = re.match(r"\[?\(([\w]+), \[([\d., ]+)\]\)", line.strip())
        match = re.match(r"\[?\(?'?([\w\/.]+)'?, \[([\d, ]+)\]\)?,?", line.strip())

        if match:
            prompt = match.group(1)
            values = list(map(float, match.group(2).split(',')))

            min_x, min_y, min_z, box_width, box_height, box_depth = values
            max_x = min_x + box_width
            max_y = min_y + box_height
            max_z = min_z + box_depth
            original_data.append({
                'index': i,
                'prompt': prompt,
                'aabb': [min_x, min_y, min_z, max_x, max_y, max_z]
            })

    # Get extreme coordinates from the original data
    min_x = min([item['aabb'][0] for item in original_data])
    max_x = max([item['aabb'][3] for item in original_data])
    min_y = min([item['aabb'][1] for item in original_data])
    max_y = max([item['aabb'][4] for item in original_data])
    min_z = min([item['aabb'][2] for item in original_data])
    max_z = max([item['aabb'][5] for item in original_data])

    def map_to_range(value, original_min, original_max, target_min, target_max):
        return ((value - original_min) / (original_max - original_min)) * (target_max - target_min) + target_min

    result = []

    for item in original_data:

        if fixed_mapping:

            mapped_min_x = map_to_range(item['aabb'][0], 0, 512, *target_range)
            mapped_max_x = map_to_range(item['aabb'][3], 0, 512, *target_range)
            mapped_min_y = map_to_range(item['aabb'][1], 0, 512, *target_range)
            mapped_max_y = map_to_range(item['aabb'][4], 0, 512, *target_range)
            mapped_min_z = map_to_range(item['aabb'][2], 0, 512, *target_range)
            mapped_max_z = map_to_range(item['aabb'][5], 0, 512, *target_range)
        else:
            max_val = max(max_x, max_y, max_z)
            min_val = min(min_x, min_y, min_z)
            # mapped_min_x = map_to_range(item['aabb'][0], min_x, max_x, *target_range)
            # mapped_max_x = map_to_range(item['aabb'][3], min_x, max_x, *target_range)
            # mapped_min_y = map_to_range(item['aabb'][1], min_y, max_y, *target_range)
            # mapped_max_y = map_to_range(item['aabb'][4], min_y, max_y, *target_range)
            # mapped_min_z = map_to_range(item['aabb'][2], min_z, max_z, *target_range)
            # mapped_max_z = map_to_range(item['aabb'][5], min_z, max_z, *target_range)

            mapped_min_x = map_to_range(item['aabb'][0], min_val, max_val, *target_range)
            mapped_max_x = map_to_range(item['aabb'][3], min_val, max_val, *target_range)
            mapped_min_y = map_to_range(item['aabb'][1], min_val, max_val, *target_range)
            mapped_max_y = map_to_range(item['aabb'][4], min_val, max_val, *target_range)
            mapped_min_z = map_to_range(item['aabb'][2], min_val, max_val, *target_range)
            mapped_max_z = map_to_range(item['aabb'][5], min_val, max_val, *target_range)

        result.append({
            'index': item['index'],
            'prompt': item['prompt'],
            'aabb': torch.FloatTensor([mapped_min_x, mapped_min_y, mapped_min_z, mapped_max_x, mapped_max_y, mapped_max_z])
        })

    if return_full_prompt:
        prompt =  os.path.splitext(os.path.basename(file_path))[0].replace('_', ' ')
        return result, prompt
    else:
        return result


def process_gpt_file(file_path, target_range = [-2, 2], fixed_mapping = True, return_full_prompt = False):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    original_data = []
    for i, line in enumerate(lines):
        match = re.match(r"\[?\('([\w\s]+)', \[([\d., ]+)\]\)", line.strip())
        if match:
            prompt = match.group(1)
            values = list(map(float, match.group(2).split(',')))

            min_x, min_y, min_z, box_width, box_height, box_depth = values
            max_x = min_x + box_width
            max_y = min_y + box_height
            max_z = min_z + box_depth
            original_data.append({
                'index': i,
                'prompt': prompt,
                'aabb': [min_x, min_y, min_z, max_x, max_y, max_z]
            })

    # Get extreme coordinates from the original data
    min_x = min([item['aabb'][0] for item in original_data])
    max_x = max([item['aabb'][3] for item in original_data])
    min_y = min([item['aabb'][1] for item in original_data])
    max_y = max([item['aabb'][4] for item in original_data])
    min_z = min([item['aabb'][2] for item in original_data])
    max_z = max([item['aabb'][5] for item in original_data])

    def map_to_range(value, original_min, original_max, target_min, target_max):
        return ((value - original_min) / (original_max - original_min)) * (target_max - target_min) + target_min

    result = []

    for item in original_data:

        if fixed_mapping:

            mapped_min_x = map_to_range(item['aabb'][0], 0, 512, *target_range)
            mapped_max_x = map_to_range(item['aabb'][3], 0, 512, *target_range)
            mapped_min_y = map_to_range(item['aabb'][1], 0, 512, *target_range)
            mapped_max_y = map_to_range(item['aabb'][4], 0, 512, *target_range)
            mapped_min_z = map_to_range(item['aabb'][2], 0, 512, *target_range)
            mapped_max_z = map_to_range(item['aabb'][5], 0, 512, *target_range)
        else:
            max_val = max(max_x, max_y, max_z)
            min_val = min(min_x, min_y, min_z)
            # mapped_min_x = map_to_range(item['aabb'][0], min_x, max_x, *target_range)
            # mapped_max_x = map_to_range(item['aabb'][3], min_x, max_x, *target_range)
            # mapped_min_y = map_to_range(item['aabb'][1], min_y, max_y, *target_range)
            # mapped_max_y = map_to_range(item['aabb'][4], min_y, max_y, *target_range)
            # mapped_min_z = map_to_range(item['aabb'][2], min_z, max_z, *target_range)
            # mapped_max_z = map_to_range(item['aabb'][5], min_z, max_z, *target_range)

            mapped_min_x = map_to_range(item['aabb'][0], min_val, max_val, *target_range)
            mapped_max_x = map_to_range(item['aabb'][3], min_val, max_val, *target_range)
            mapped_min_y = map_to_range(item['aabb'][1], min_val, max_val, *target_range)
            mapped_max_y = map_to_range(item['aabb'][4], min_val, max_val, *target_range)
            mapped_min_z = map_to_range(item['aabb'][2], min_val, max_val, *target_range)
            mapped_max_z = map_to_range(item['aabb'][5], min_val, max_val, *target_range)

        result.append({
            'index': item['index'],
            'prompt': item['prompt'],
            'aabb': torch.FloatTensor([mapped_min_x, mapped_min_y, mapped_min_z, mapped_max_x, mapped_max_y, mapped_max_z])
        })

    if return_full_prompt:
        prompt =  os.path.splitext(os.path.basename(file_path))[0].replace('_', ' ')
        return result, prompt
    else:
        return result


def import_all_modules(folder_path, self_module_name=None):
    for filename in os.listdir(folder_path):
        if filename.endswith('.py') and filename != '__init__.py' and (self_module_name not in filename):
            file_path = os.path.join(folder_path, filename)
            module_name = os.path.splitext(filename)[0]
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)


def load_zero123_images(image_path, width, height, rank, requires_depth=False, requires_normal=False):
    # Check if image exists
    assert os.path.exists(image_path), f"Could not find image {image_path}!"

    # Load and process RGBA image
    rgba = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
    rgba = cv2.resize(rgba, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

    # Split into RGB and alpha mask
    rgb = rgba[..., :3]
    mask = rgba[..., 3:] > 0.5

    # Convert to tensors and move to specified rank
    rgb_tensor = torch.from_numpy(rgb).unsqueeze(0).contiguous().to(rank)
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(rank)

    print(f"[INFO] single image dataset: load image {image_path} {rgb_tensor.shape}")

    # Initialize depth and normal tensors
    depth_tensor = None
    normal_tensor = None

    # Load depth data if required
    if requires_depth:
        depth_path = image_path.replace("_rgba.png", "_depth.png")
        assert os.path.exists(depth_path), f"Could not find depth image {depth_path}!"
        
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_AREA)
        depth_tensor = torch.from_numpy(depth.astype(np.float32) / 255.0).unsqueeze(0).to(rank)

        print(f"[INFO] single image dataset: load depth {depth_path} {depth_tensor.shape}")

    # Load normal data if required
    if requires_normal:
        normal_path = image_path.replace("_rgba.png", "_normal.png")
        assert os.path.exists(normal_path), f"Could not find normal image {normal_path}!"
        
        normal = cv2.imread(normal_path, cv2.IMREAD_UNCHANGED)
        normal = cv2.resize(normal, (width, height), interpolation=cv2.INTER_AREA)
        normal_tensor = torch.from_numpy(normal.astype(np.float32) / 255.0).unsqueeze(0).to(rank)

        print(f"[INFO] single image dataset: load normal {normal_path} {normal_tensor.shape}")

    return rgb_tensor, mask_tensor, depth_tensor, normal_tensor