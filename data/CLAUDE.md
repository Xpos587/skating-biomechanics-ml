# data/CLAUDE.md — Data Files

## Structure

```
data/
├── DATASETS.md                        # Dataset registry and relationships
├── datasets/                          # Downloaded ML datasets
│   ├── athletepose3d/                 # AthletePose3D (71GB, 3D poses, 12 sports)
│   ├── figure-skating-classification/ # 5168 sequences, 64 element classes
│   └── mmfs/                          # MMFS (26198 sequences, 256 categories)
├── models/                            # ML model weights
├── references/                        # Reference motion .npz files for comparison
├── raw/                               # Raw video files + transcripts
├── processed/                         # Processed/normalized data
└── uploads/                           # User upload staging (not in git)
```

## Datasets

| Dataset | Content | Size | Status |
|---------|---------|------|--------|
| AthletePose3D | 1.3M frames, 12 sports, 3D poses | 71GB | Downloaded |
| Figure-Skating-Classification | 5168 sequences, 64 classes | 340MB | Downloaded |
| MMFS | 26198 skeleton sequences, 256 categories | 1.7GB | Downloaded |

See `DATASETS.md` for full registry, download links, and inter-dataset relationships.

## Notes

- `uploads/` is for local dev only — production uploads go to Cloudflare R2
- `models/` contains ONNX model weights (not in git — use `ml/scripts/download_ml_models.py`)
- `references/` stores `.npz` files built with `ml/scripts/build_references.py`
- Large files (videos, model weights) are in `.gitignore`
