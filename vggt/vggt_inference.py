# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import glob
import time
import threading
import argparse
from typing import List, Optional

import numpy as np
import torch
from tqdm.auto import tqdm
import viser
import viser.transforms as viser_tf
import cv2


try:
    import onnxruntime
except ImportError:
    print("onnxruntime not found. Sky segmentation may not work.")

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.visual_util import segment_sky, download_file_from_url
from jhutil import cache_output

device = "cuda" if torch.cuda.is_available() else "cpu"
model = None

def load_model():
    global model
    if model is None:
        model = VGGT()
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        model.load_state_dict(torch.hub.load_state_dict_from_url(_URL, model_dir="/root/data1/jinhyeok/checkpoints/vggt"))
        model.eval()
        model = model.to(device)
    return model


def unload_model():
    global model
    if model is not None:
        model.cpu()
        del model
        torch.cuda.empty_cache()


@cache_output(func_name="vggt_inference")
def vggt_inference(image_folder: str, n_images: int = -1) -> dict:
    if model is None:
        load_model()
        
    # Use the provided image folder path
    print(f"Loading images from {image_folder}...")
    image_names = glob.glob(os.path.join(image_folder, "*"))
    
    if n_images > 0 and n_images < len(image_names):
        image_indices = np.linspace(0, len(image_names) - 1, n_images).astype(int)
        image_names = [image_names[i] for i in image_indices]
        

    print(f"Found {len(image_names)} images")

    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {images.shape}")

    print("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    print("Processing model outputs...")

    return predictions

if __name__ == "__main__":
    main()
