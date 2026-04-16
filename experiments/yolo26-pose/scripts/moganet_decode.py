import sys
sys.path.insert(0, "/root")

import cv2
import numpy as np
import torch
import json
import pycocotools.mask as cocomask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Mock xtcocotools
class XtcocoTools:
    class coco:
        COCO = COCO
    class cocoeval:
        COCOeval = COCOeval
    class mask:
        encode = cocomask.encode
        decode = cocomask.decode

sys.modules["xtcocotools"] = XtcocoTools
sys.modules["xtcocotools.coco"] = XtcocoTools.coco
sys.modules["xtcocotools.cocoeval"] = XtcocoTools.cocoeval
sys.modules["xtcocotools.mask"] = XtcocoTools.mask

from moganet_official import MogaNet_feat

# ============ MMPose UDP Decoder Functions ============

def gaussian_blur(heatmaps: np.ndarray, kernel: int = 11) -> np.ndarray:
    """Modulate heatmap distribution with Gaussian (DarkPose)."""
    assert kernel % 2 == 1
    border = (kernel - 1) // 2
    K, H, W = heatmaps.shape
    for k in range(K):
        origin_max = np.max(heatmaps[k])
        dr = np.zeros((H + 2 * border, W + 2 * border), dtype=np.float32)
        dr[border:-border, border:-border] = heatmaps[k].copy()
        dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
        heatmaps[k] = dr[border:-border, border:-border].copy()
        heatmaps[k] *= origin_max / np.max(heatmaps[k])
    return heatmaps


def get_heatmap_maximum(heatmaps: np.ndarray):
    """Get maximum response location and value from heatmaps."""
    K, H, W = heatmaps.shape
    heatmaps_flatten = heatmaps.reshape(K, -1)
    y_locs, x_locs = np.unravel_index(
        np.argmax(heatmaps_flatten, axis=1), shape=(H, W))
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    vals = np.amax(heatmaps_flatten, axis=1)
    locs[vals <= 0.] = -1
    return locs, vals


def refine_keypoints_dark_udp(keypoints: np.ndarray, heatmaps: np.ndarray,
                              blur_kernel_size: int) -> np.ndarray:
    """Refine keypoints using UDP DarkPose algorithm."""
    N, K = keypoints.shape[:2]
    H, W = heatmaps.shape[1:]
    
    # Modulate heatmaps
    heatmaps = gaussian_blur(heatmaps, blur_kernel_size)
    np.clip(heatmaps, 1e-3, 50., heatmaps)
    np.log(heatmaps, heatmaps)
    
    heatmaps_pad = np.pad(
        heatmaps, ((0, 0), (1, 1), (1, 1)), mode='edge').flatten()
    
    for n in range(N):
        index = keypoints[n, :, 0] + 1 + (keypoints[n, :, 1] + 1) * (W + 2)
        index += (W + 2) * (H + 2) * np.arange(0, K)
        index = index.astype(int).reshape(-1, 1)
        i_ = heatmaps_pad[index]
        ix1 = heatmaps_pad[index + 1]
        iy1 = heatmaps_pad[index + W + 2]
        ix1y1 = heatmaps_pad[index + W + 3]
        ix1_y1_ = heatmaps_pad[index - W - 3]
        ix1_ = heatmaps_pad[index - 1]
        iy1_ = heatmaps_pad[index - 2 - W]
        
        dx = 0.5 * (ix1 - ix1_)
        dy = 0.5 * (iy1 - iy1_)
        derivative = np.concatenate([dx, dy], axis=1)
        derivative = derivative.reshape(K, 2, 1)
        
        dxx = ix1 - 2 * i_ + ix1_
        dyy = iy1 - 2 * i_ + iy1_
        dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
        hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1)
        hessian = hessian.reshape(K, 2, 2)
        hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
        keypoints[n] -= np.einsum('imn,ink->imk', hessian,
                                  derivative).squeeze()
    return keypoints


# ============ DeconvHead ============

class DeconvHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True)
        )
        self.deconv2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True)
        )
        self.deconv3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True)
        )
        self.final = torch.nn.Conv2d(256, 17, 1)
    
    def forward(self, x):
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.final(x)
        return x


def decode_udp_heatmap(heatmaps: np.ndarray, input_size: tuple, 
                       heatmap_size: tuple, blur_kernel_size: int = 11):
    """Decode UDP heatmaps to keypoints using DarkPose refinement."""
    # Get initial keypoints from heatmap maximum
    keypoints, scores = get_heatmap_maximum(heatmaps)
    
    # Add instance dimension
    keypoints = keypoints[None]
    scores = scores[None]
    
    # Refine with DarkPose UDP
    keypoints = refine_keypoints_dark_udp(keypoints, heatmaps, blur_kernel_size)
    
    # Normalize to input image coordinates
    W, H = heatmap_size
    keypoints = keypoints / [W - 1, H - 1] * input_size
    
    return keypoints, scores


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    backbone = MogaNet_feat(arch="base", out_indices=(0,1,2,3)).to(device)
    head = DeconvHead().to(device)
    
    # Load checkpoint
    ckpt = torch.load("/root/data/models/athletepose3d/moganet_b_ap2d_384x288.pth", 
                      weights_only=False, map_location=device)
    
    # Load weights
    backbone_state = {k.replace("backbone.", ""): v 
                     for k, v in ckpt["state_dict"].items() if k.startswith("backbone.")}
    backbone.load_state_dict(backbone_state, strict=True)
    backbone.eval()
    
    head_state = {}
    for k, v in ckpt["state_dict"].items():
        if "keypoint_head" in k:
            if "deconv_layers" in k:
                parts = k.replace("keypoint_head.deconv_layers.", "").split(".")
                idx, rest = int(parts[0]), ".".join(parts[1:])
                mapping = {0: "deconv1.0", 1: "deconv1.1", 3: "deconv2.0", 
                          4: "deconv2.1", 6: "deconv3.0", 7: "deconv3.1"}
                if idx in mapping:
                    head_state[f"{mapping[idx]}.{rest}"] = v
            elif "final_layer" in k:
                head_state[k.replace("keypoint_head.final_layer", "final")] = v
    head.load_state_dict(head_state, strict=True)
    head.eval()
    
    # Image paths
    val_images = [
        "/root/data/datasets/athletepose3d/pose_2d/test_set/00000000160.jpg",
        "/root/data/datasets/athletepose3d/pose_2d/test_set/00000000161.jpg",
        "/root/data/datasets/athletepose3d/pose_2d/test_set/00000000103.jpg"
    ]
    
    # Normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    # Config from MogaNet COCO config
    input_size = (384, 288)  # (width, height)
    heatmap_size = (96, 72)  # (width, height)
    blur_kernel_size = 11
    
    # COCO skeleton
    skeleton = [
        [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
        [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
        [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
        [1, 3], [2, 4], [3, 5], [4, 6]
    ]
    
    results_info = []
    
    with torch.no_grad():
        for i, img_path in enumerate(val_images[:3]):
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load {img_path}")
                continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (384, 288))
            
            # Normalize
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
            img_tensor = (img_tensor - torch.from_numpy(mean).view(3, 1, 1)) / torch.from_numpy(std).view(3, 1, 1)
            img_tensor = img_tensor.unsqueeze(0).to(device)
            
            # Forward
            features = backbone(img_tensor)
            heatmaps = head(features[3])  # (1, 17, 72, 96)
            
            # Convert to numpy and transpose to (K, H, W)
            heatmaps_np = heatmaps[0].cpu().numpy()  # (17, 72, 96)
            
            # Decode with UDP DarkPose
            keypoints, scores = decode_udp_heatmap(
                heatmaps_np, input_size, heatmap_size, blur_kernel_size
            )
            
            # Normalize to 0-1
            keypoints_norm = keypoints / [384, 288]
            
            # Draw skeleton
            img_draw = img.copy()
            h, w = img.shape[:2]
            
            for pair in skeleton:
                kp1 = keypoints_norm[0, pair[0]]
                kp2 = keypoints_norm[0, pair[1]]
                if kp1[0] > 0 and kp1[1] > 0 and kp2[0] > 0 and kp2[1] > 0:
                    pt1 = (int(kp1[0] * w), int(kp1[1] * h))
                    pt2 = (int(kp2[0] * w), int(kp2[1] * h))
                    cv2.line(img_draw, pt1, pt2, (0, 255, 0), 2)
            
            for kp in keypoints_norm[0]:
                if kp[0] > 0 and kp[1] > 0:
                    pt = (int(kp[0] * w), int(kp[1] * h))
                    cv2.circle(img_draw, pt, 4, (0, 0, 255), -1)
            
            # Save
            out_path = f"/root/moganet_udp_{i}.jpg"
            cv2.imwrite(out_path, img_draw)
            print(f"Saved {out_path}")
            
            results_info.append({
                "frame": i,
                "image": img_path,
                "keypoints": keypoints_norm[0].tolist()
            })
    
    # Save results
    with open("/root/moganet_udp_results.json", "w") as f:
        json.dump(results_info, f, indent=2)
    
    for info in results_info:
        print(f"Frame {info['frame']}: {info['image'].split('/')[-1]}")

if __name__ == "__main__":
    main()
