
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

# Use rtmlib directly
from rtmlib import Body

# RTMO setup
detector = Body(
    det="rtmo",
    pose="rtmo",
    backend="onnxruntime",
    device="cuda"
)

frames_dir = Path("/root/data/datasets/skatingverse/frames")
output_file = Path("/root/data/datasets/skatingverse_pseudo/labels_rtmo.json")
frame_dirs = sorted([d for d in frames_dir.iterdir() if d.is_dir()])

results = []
for video_dir in tqdm(frame_dirs[:100], desc="Processing"):
    for frame_path in sorted(video_dir.glob("*.jpg")):
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        
        # Run RTMO
        keypoints, scores = detector(img)
        
        if len(keypoints) == 0:
            continue
        
        # Take first person
        kp = keypoints[0]  # (17, 2)
        sc = scores[0] if len(scores) > 0 else np.zeros(17)
        
        keypoints_flat = []
        for i in range(17):
            x, y = kp[i]
            conf = float(sc[i]) if i < len(sc) else 0.5
            if conf > 0.3 and x > 0 and y > 0:
                keypoints_flat.extend([float(x), float(y), 2.0])
            elif conf > 0.1:
                keypoints_flat.extend([float(x), float(y), 1.0])
            else:
                keypoints_flat.extend([0, 0, 0])
        
        results.append({
            "image_id": str(frame_path.relative_to(frames_dir)),
            "video_id": video_dir.name,
            "keypoints": keypoints_flat,
            "score": float(np.mean(sc))
        })

output = {"images": len(results), "annotations": results}
with open(output_file, "w") as f:
    json.dump(output, f)
print(f"Saved {len(results)} poses to {output_file}")
