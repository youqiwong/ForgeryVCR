# Standalone NPP service bundled with ForgeryVCR.
# Module for pre-loading and running inference with the Noiseprint++ model.

import sys
import os
import numpy as np
import cv2
import torch

# 确保能找到项目内的模块。
# 假设此文件与 'dataset' 和 'lib' 目录在同一项目结构下。
# 如果不是，你可能需要调整 sys.path。
path = os.path.dirname(os.path.realpath(__file__))
if path not in sys.path:
    sys.path.insert(0, path)

try:
    from dataset.dataset_test import TestDataset
    from lib.models.DnCNN import make_net
except ImportError as e:
    raise ImportError(f"Could not import NPP modules. Ensure this service file is in the correct directory. Original error: {e}")


def load_npp_model(model_path: str, device: torch.device):
    """
    Loads the Noiseprint++ model from disk into memory (GPU or CPU).
    This function should be called only once when the server starts.

    Args:
        model_path (str): Path to the .th model file.
        device (torch.device): The device to load the model onto.

    Returns:
        torch.nn.Module: The loaded and initialized model.
    """
    print(f"=> [NPP Service] Loading Noiseprint++ extractor from {model_path} onto {device}")
    
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    num_levels = 17
    extractor_model = make_net(
        nplanes_in=3,
        kernels=[3, ] * num_levels,
        features=[64, ] * (num_levels - 1) + [1, ],
        bns=[False, ] + [True, ] * (num_levels - 2) + [False, ],
        acts=['relu', ] * (num_levels - 1) + ['linear', ],
        dilats=[1, ] * num_levels,
        bn_momentum=0.1,
        padding=1
    )

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['network']
    extractor_model.load_state_dict(state_dict)
    extractor_model = extractor_model.to(device)
    extractor_model.eval()
    
    print("=> [NPP Service] Model loaded successfully and set to evaluation mode.")
    return extractor_model


def run_npp_inference(model: torch.nn.Module, image_path: str, device: torch.device):
    """
    Runs inference on a single image using a pre-loaded model.

    Args:
        model (torch.nn.Module): The pre-loaded Noiseprint++ model.
        image_path (str): Path to the input image.
        device (torch.device): The device the model is on.

    Returns:
        np.ndarray: An 8-bit visual representation of the NPP map, or None if failed.
    """
    try:
        # --- Data Loading for a single image ---
        # The original script's TestDataset is efficient for this.
        test_dataset = TestDataset(list_img=[image_path])
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=0) # num_workers=0 for simplicity in server env

        with torch.no_grad():
            for rgb_tensor, _ in testloader:
                # --- Inference ---
                rgb_tensor = rgb_tensor.to(device)
                npp = model(rgb_tensor)
                npp_array = torch.squeeze(npp).cpu().numpy()

                # --- Enhanced Visualization Logic (copied from original script) ---
                mean, std = npp_array.mean(), npp_array.std()
                vmin = mean - 3 * std
                vmax = mean + 3 * std
                clipped_npp = np.clip(npp_array, vmin, vmax)
                
                if vmax - vmin > 1e-6:
                    npp_visual = 255 * (clipped_npp - vmin) / (vmax - vmin)
                else:
                    npp_visual = np.full_like(clipped_npp, 128)
                
                # Convert to an 8-bit unsigned integer image and return
                return npp_visual.astype(np.uint8)

    except Exception as e:
        print(f"\n[NPP Service] Error during inference for {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Should not be reached if an image is found
    return None
