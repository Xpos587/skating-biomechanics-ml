import json
import cv2
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

val_labels_dir = Path('/root/data/datasets/yolo26_ap3d/labels/val')
val_images_dir = Path('/root/data/datasets/yolo26_ap3d/images/val')

coco_gt = {'images': [], 'annotations': [], 'categories': [{'id': 1, 'name': 'person'}]}

ann_id = 0
for img_path in sorted(val_images_dir.glob('*.jpg'))[:100]:
    img_id = len(coco_gt['images'])
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    
    coco_gt['images'].append({
        'id': img_id,
        'file_name': img_path.name,
        'height': h,
        'width': w
    })
    
    label_path = val_labels_dir / (img_path.stem + '.txt')
    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 55:  # class + bbox(4) + keypoints(51)
                    continue
                
                try:
                    vals = list(map(float, parts))
                    # YOLO format: class xc yc w h ... keypoints
                    xc, yc, bw, bh = vals[1:5]
                    kpts = vals[5:56]  # 17 * 3 = 51 values
                    
                    # Convert to COCO bbox format (x, y, w, h in pixels)
                    x1 = (xc - bw/2) * w
                    y1 = (yc - bh/2) * h
                    bbox_w = bw * w
                    bbox_h = bh * h
                    
                    coco_gt['annotations'].append({
                        'id': ann_id,
                        'image_id': img_id,
                        'category_id': 1,
                        'bbox': [x1, y1, bbox_w, bbox_h],
                        'keypoints': kpts,
                        'num_keypoints': 17,
                        'area': bbox_w * bbox_h,
                        'iscrowd': 0
                    })
                    ann_id += 1
                except:
                    pass

with open('/root/rtmo_val_gt.json', 'w') as f:
    json.dump(coco_gt, f)

with open('/root/rtmo_val_predictions.json') as f:
    coco_dt = json.load(f)

coco_gt_obj = COCO('/root/rtmo_val_gt.json')
coco_dt_obj = coco_gt_obj.loadRes('/root/rtmo_val_predictions.json')

coco_eval = COCOeval(coco_gt_obj, coco_dt_obj, 'keypoints')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

print(f'\nRTMO-s mAP50-95: {coco_eval.stats[0]:.3f}')
print(f'RTMO-s mAP50: {coco_eval.stats[1]:.3f}')
