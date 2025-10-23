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


@cache_output(func_name="_vggt_inference", override=False)
def _vggt_inference(image_names: list=None, precision=torch.float32) -> dict:

    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {images.shape}")

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=precision, enabled=False):
            predictions = model(images)

    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].squeeze(0)
    
    print("Processing model outputs...")

    return predictions


def vggt_inference(image_folder: str=None, image_names: list=None, n_images: int = -1, precision=torch.float32) -> dict:
    if model is None:
        load_model()
        
    # Use the provided image folder path
    print(f"Loading images from {image_folder}...")
    if image_names is None:
        image_names = glob.glob(os.path.join(image_folder, "*"))
        try:
            image_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        except:
            image_names.sort(key=lambda p: os.path.splitext(p)[0])
    
    if n_images > 0 and n_images < len(image_names):
        image_indices = np.linspace(0, len(image_names) - 1, n_images).astype(int)
        image_names = [image_names[i] for i in image_indices]
    
    print(f"Found {len(image_names)} images")
    return _vggt_inference(image_names, precision)
        


if __name__ == "__main__":
    main()
