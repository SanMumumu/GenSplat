# GenSplat

This repository keeps three feed-forward Gaussian reconstruction variants under one training pipeline so they can be trained and compared directly.

| Variant | Student | Teacher / pseudo-supervision | Config name |
| --- | --- | --- | --- |
| AnySplat | VGGT | VGGT | `anysplat` |
| GenSplat | VGGT | Pi3 | `gensplat` |
| GenSplatPi3 | Pi3 | Pi3 | `gensplatpi3` |

Source layout:

- `src/model/encoder/anysplat.py`: preserved `VGGT student + VGGT teacher`
- `src/model/encoder/gensplat.py`: main `VGGT student + Pi3 teacher`
- `src/model/encoder/gensplatpi3.py`: comparison `Pi3 student + Pi3 teacher`

## 0. Environment Setup

This repository follows the same environment style as [AnySplat](https://github.com/InternRobotics/AnySplat), with one extra dependency required by the Pi3-based paths: `utils3d`.

The commands below are a practical Linux baseline for Python 3.10, PyTorch 2.2, and CUDA 12.1.

```bash
conda create -y -n gensplat python=3.10
conda activate gensplat

pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

pip install \
  numpy==1.25.0 wheel tqdm lightning black ruff hydra-core jaxtyping beartype \
  einops colorama scikit-image colorspacious matplotlib moviepy imageio timm \
  dacite lpips e3nn plyfile tabulate svg.py scikit-video swanlab tyro viser \
  pycolmap opencv-python Pillow huggingface_hub gradio xformers==0.0.24 \
  torch_scatter==2.1.2 pydantic open3d safetensors utils3d

pip install git+https://github.com/facebookresearch/pytorch3d.git
pip install https://github.com/nerfstudio-project/gsplat/releases/download/v1.4.0/gsplat-1.4.0%2Bpt22cu121-cp310-cp310-linux_x86_64.whl
```

Notes:

- If your CUDA or PyTorch version differs, adjust the `torch`, `xformers`, `torch_scatter`, and `gsplat` wheels accordingly.
- The training entry point is `src/main.py`.
- The default training preset in this repository is `+experiment=droid`.

## 1. Training the Three Variants

### 1.1 Common Preparation

Before launching training:

1. Set the dataset root in `config/dataset/droid.yaml`, or override it from the command line.
2. Decide whether you want to initialize from an existing checkpoint or start from the default Hugging Face weights.
3. Be explicit about `checkpointing.load`, because `config/experiment/droid.yaml` contains a sample checkpoint path.

Useful checkpoint arguments:

- `model.encoder.pretrained_weights`: initialize the student or the full model from a local checkpoint
- `model.encoder.distiller_pretrained_weights`: initialize the Pi3 teacher in the `gensplat` variant
- `checkpointing.load`: resume or override from an existing full checkpoint

If you do not want to resume from an existing run, set:

```bash
checkpointing.load=null
```

### 1.2 AnySplat: VGGT Student + VGGT Teacher

This is the preserved original baseline.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py \
  +experiment=droid \
  model/encoder=anysplat \
  checkpointing.load=null \
  model.encoder.pretrained_weights=/path/to/anysplat_model.safetensors \
  swanlab.mode=offline
```

If you want to start from the default VGGT Hugging Face weights only, omit `model.encoder.pretrained_weights`.

### 1.3 GenSplat: VGGT Student + Pi3 Teacher

This is the main variant in this repository.

Recommended initialization:

- student initialized from an AnySplat checkpoint
- teacher initialized from a Pi3 checkpoint

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py \
  +experiment=droid \
  model/encoder=gensplat \
  checkpointing.load=null \
  model.encoder.pretrained_weights=/path/to/anysplat_model.safetensors \
  model.encoder.distiller_pretrained_weights=/path/to/pi3_checkpoint.pt \
  swanlab.mode=offline
```

If `model.encoder.distiller_pretrained_weights` is omitted, the teacher falls back to `yyfz233/Pi3` through `Pi3.from_pretrained(...)`.

### 1.4 GenSplatPi3: Pi3 Student + Pi3 Teacher

This variant is kept as a comparison model.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py \
  +experiment=droid \
  model/encoder=gensplatpi3 \
  checkpointing.load=null \
  swanlab.mode=offline
```

You can still provide `model.encoder.pretrained_weights` or `checkpointing.load` if you have a compatible full checkpoint.

### 1.5 Output Directory

With `+experiment=droid`, training outputs are written under:

```text
output/exp_<swanlab.name>/<timestamp>/
```

The default checkpoint directory is:

```text
output/exp_<swanlab.name>/<timestamp>/checkpoints/
```

## 2. Using `augment_parquet.py`

The repository includes `augment_parquet.py` for parquet-based data augmentation with single-view GenSplat novel-view synthesis.

The script:

- reads all `.parquet` files from `--input_dir`
- processes `observation.image1` and `observation.image2` independently
- reconstructs each image as a single-view Gaussian scene
- perturbs the predicted camera translation by `--trans_noise`
- renders a new view
- writes the rendered PNG bytes back to the parquet rows
- keeps `observation.image3` unchanged

### 2.1 Example Command

```bash
python augment_parquet.py \
  --input_dir /data/chunk-000 \
  --output_dir /data/chunk-000-new \
  --ckpt /path/to/gensplat_hf_export \
  --trans_noise 0.02
```

Arguments:

- `--input_dir`: directory containing input parquet files
- `--output_dir`: directory for augmented parquet files; defaults to `<input_dir>-new`
- `--ckpt`: model path or Hugging Face identifier accepted by `GenSplat.from_pretrained(...)`
- `--trans_noise`: translation noise magnitude applied to the predicted pose before rendering

### 2.2 Input Assumptions

The script currently assumes each parquet row contains these columns:

- `observation.image1`
- `observation.image2`
- `observation.image3`

It also assumes each of these columns is dict-like and contains at least:

- `["bytes"]`: encoded image bytes

The script center-crops each processed image so that height and width are multiples of `14`, which matches the model patch size.

## 3. `augment_parquet.py` Output Structure

The output directory mirrors the input directory at the file level:

```text
input_dir/
  part-000.parquet
  part-001.parquet

output_dir/
  part-000.parquet
  part-001.parquet
```

For every output parquet file:

- the file name is unchanged
- the row count is unchanged
- all non-image columns are preserved
- `observation.image3` is preserved
- `observation.image1["bytes"]` is replaced by rendered PNG bytes
- `observation.image2["bytes"]` is replaced by rendered PNG bytes

The schema is preserved; only the payload inside the selected image columns is updated.

### 3.1 Per-Row Behavior

Each row keeps the same top-level layout:

```text
row
|- observation.image1
|  \- bytes   <- replaced
|- observation.image2
|  \- bytes   <- replaced
\- observation.image3
   \- bytes   <- unchanged
```

If `observation.image1` or `observation.image2` contains additional metadata keys, the script keeps that dictionary and only overwrites the `bytes` field.

## 4. Recommended Workflow

For a clean comparison:

1. Train `anysplat` as the preserved baseline.
2. Train `gensplat` with the AnySplat student checkpoint and the Pi3 teacher checkpoint.
3. Train `gensplatpi3` as the Pi3-only comparison model.
4. Run `augment_parquet.py` only after you have a trained `gensplat` checkpoint exported in a format compatible with `GenSplat.from_pretrained(...)`.

## 5. Relevant Files

- `src/main.py`: training entry point
- `config/experiment/droid.yaml`: main training preset
- `config/model/encoder/anysplat.yaml`: AnySplat encoder config
- `config/model/encoder/gensplat.yaml`: GenSplat encoder config
- `config/model/encoder/gensplatpi3.yaml`: GenSplatPi3 encoder config
- `augment_parquet.py`: parquet augmentation utility
