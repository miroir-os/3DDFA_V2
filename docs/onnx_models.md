# ONNX models

This project ships three ONNX graphs (face detector, dense BFM decoder, TDDFA
regressor) plus optional pose-baked variants of the regressor. They are all
loaded by `onnxruntime` at runtime, with no PyTorch on the inference path.

## Where the files live

| File | Path | Source |
| :-- | :-- | :-- |
| FaceBoxes detector | `FaceBoxes/weights/FaceBoxesProd.onnx` | converted from `FaceBoxesProd.pth` |
| BFM dense decoder | `configs/bfm_noneck_v3.onnx` | converted from `configs/bfm_noneck_v3.pkl` |
| TDDFA regressor (mb1) | `weights/mb1_120x120.onnx` | converted from `weights/mb1_120x120.pth` |
| TDDFA regressor (mb05) | `weights/mb05_120x120.onnx` | converted from `weights/mb05_120x120.pth` |
| TDDFA regressor (resnet22) | `weights/resnet22.onnx` | converted from `weights/resnet22.pth` |
| TDDFA pose-baked (mb1) | `weights/mb1_120x120_pose.onnx` | baked from mb1 + `param_mean_std_62d_120x120.pkl` via `bake_pose_onnx.py` |
| TDDFA pose-baked (mb05) | `weights/mb05_120x120_pose.onnx` | same, smaller backbone |
| TDDFA pose-baked (resnet22) | `weights/resnet22_pose.onnx` | same, resnet22 backbone |

The accompanying `.pth` checkpoints, `bfm_noneck_v3.pkl`, and
`param_mean_std_62d_120x120.pkl` are fetched per the links in
`weights/readme.md`, `FaceBoxes/weights/readme.md`, and `bfm/readme.md`.

## What each model does

- **FaceBoxesProd.onnx**. Single-stage face detector. Takes an RGB image at
  arbitrary resolution (dynamic batch, height, width) and returns per-anchor
  box regressions plus class scores. Wrapped by `FaceBoxes/FaceBoxes_ONNX.py`,
  which decodes the priors and runs NMS to emit `[xmin, ymin, xmax, ymax, score]`
  bboxes.

- **bfm_noneck_v3.onnx**. Dense Basel Face Model decoder. Takes
  `(R, offset, alpha_shp, alpha_exp)` and returns the 3D dense vertex tensor
  (~38k points). Used by `TDDFA_ONNX.recon_vers(..., dense_flag=True)` for the
  full mesh; sparse 68-point reconstruction stays on numpy and does not need
  this graph.

- **mb1_120x120.onnx / mb05_120x120.onnx / resnet22.onnx**. TDDFA pose and shape
  regressors. They take a `1x3x120x120` cropped face (zero-mean, unit-scale
  preprocessed externally) and return a 62-d parameter vector
  `[12 camera + 40 shape + 10 expression]`. Output is normalized; the caller
  multiplies by `param_std` and adds `param_mean` (loaded from
  `configs/param_mean_std_62d_120x120.pkl`). MobileNet-V1 (`mb1`) is the
  default, MobileNet-V1 0.5x (`mb05`) is the speed pick, ResNet22 is the
  accuracy pick.

- **`*_pose.onnx`**. Variants of the regressors above with the 62-d
  `param_mean` / `param_std` denormalization fused into the graph. They take
  the same preprocessed `1x3x120x120` input as the standard regressor (caller
  still does `(x - 127.5) / 128`) but emit an already-denormalized 62-d
  vector, so no `param_mean_std_62d_120x120.pkl` is needed at runtime.
  Consumed by `demo_pose_only.py`, which only reads the first 12 camera
  params for Euler decoding.

## Creating the ONNX files if they are missing

`FaceBoxes_ONNX` and `TDDFA_ONNX` already auto-convert on first run if the
matching `.onnx` file is absent (`FaceBoxes/FaceBoxes_ONNX.py:51`,
`TDDFA_ONNX.py:31` and `TDDFA_ONNX.py:56`). So instantiating either class
once with the corresponding `.pth` / `.pkl` in place is enough to materialize
the file.

Manual conversion entry points:

```python
# FaceBoxesProd.pth -> FaceBoxes/weights/FaceBoxesProd.onnx
from FaceBoxes.onnx import convert_to_onnx
convert_to_onnx('FaceBoxes/weights/FaceBoxesProd.onnx')

# weights/<name>.pth -> weights/<name>.onnx (TDDFA regressor)
from utils.onnx import convert_to_onnx
convert_to_onnx(
    arch='mobilenet', widen_factor=1.0, size=120, num_params=62,
    checkpoint_fp='weights/mb1_120x120.pth',
)

# configs/bfm_noneck_v3.pkl -> configs/bfm_noneck_v3.onnx
from bfm.bfm_onnx import convert_bfm_to_onnx
convert_bfm_to_onnx('configs/bfm_noneck_v3.onnx', shape_dim=40, exp_dim=10)
```

The TDDFA exporter reads its `arch`, `widen_factor`, `size`, and `num_params`
from the matching YAML in `configs/` (e.g. `mb1_120x120.yml`), so you can
also drive it from the YAML:

```python
import yaml
from utils.onnx import convert_to_onnx
cfg = yaml.safe_load(open('configs/mb1_120x120.yml'))
convert_to_onnx(**cfg)
```

All three exporters call `torch.onnx.export(..., dynamo=False)` to keep the
TorchScript path under recent PyTorch. The TDDFA export declares a dynamic
batch dim so multi-face batching works at inference time; the FaceBoxes
export declares dynamic batch, height, and width.

## Pose-baked variants

Build them with `bake_pose_onnx.py`. The script wraps the regressor in a
module that multiplies the raw output by `param_std` and adds `param_mean`,
then exports with `dynamo=False` and a dynamic batch dim:

```bash
# Defaults: configs/mb1_120x120.yml + configs/param_mean_std_62d_120x120.pkl
#           -> weights/mb1_120x120_pose.onnx
python bake_pose_onnx.py

# Other backbones
python bake_pose_onnx.py -c configs/mb05_120x120.yml
python bake_pose_onnx.py -c configs/resnet_120x120.yml -o weights/resnet22_pose.onnx
```

`demo_pose_only.py` consumes the baked file directly. If it's missing, run
the command above with the matching YAML; the standard `mb*_120x120.onnx`
will not work because the demo skips the host-side denormalization.
