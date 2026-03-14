# NTIRE 2026 Image Denoising Challenge (noise level = 50)

This repository contains the code used to reproduce our test submission for the NTIRE 2026 Image Denoising Challenge.
Large artifacts such as `results/`, `checkpoints/`, and archived checkpoint bundles are not tracked in this repository.

## 1. Method

- Model: `MoCE_IR`
- Checkpoint: `checkpoints/MoCE_IR_AIO3/last.ckpt`
- Inference script: `src/infer_competition.py`
- Task used for submission: denoising with `sigma=50`

This submission uses the all-in-one `MoCE_IR_AIO3` checkpoint and runs it on the NTIRE 2026 testing images with original file names preserved.

## 2. Environment

We used the environment created in this repository with Python 3.10 and PyTorch 2.5.1 + CUDA 11.8.

Activate the environment:

```bash
micromamba activate moceir_pip
```

If activation is unavailable in your shell, run commands with:

```bash
micromamba run -n moceir_pip <command>
```

## 3. Data

Competition page:

```text
https://www.codabench.org/competitions/12905/#/pages-tab
```

Testing dataset download link:

```text
https://drive.usercontent.google.com/download?id=1UZA_AEdV5EgqWl9lozYo12YrET-Pno6L&export=download&authuser=0
```

In our local setup, the test images are stored at:

```text
<PATH_TO_TEST_SET>/LSDIR_DIV2K_Test_Sigma50
```

The directory contains 200 PNG images.

## 4. Checkpoint

The pretrained checkpoint is not included in this repository.

Official checkpoint source from the original MoCE-IR repository:

```text
https://drive.google.com/drive/folders/1pQBceb8cCPdIzbqbNNGqV5qNXzzqL4uK?usp=share_link
```

Expected local path after you place the checkpoint manually:

```text
checkpoints/MoCE_IR_AIO3/last.ckpt
```

## 5. Reproduce Inference

Run:

```bash
micromamba run -n moceir_pip python src/infer_competition.py \
  --model MoCE_IR \
  --checkpoint_id MoCE_IR_AIO3 \
  --input_dir <PATH_TO_TEST_SET>/LSDIR_DIV2K_Test_Sigma50 \
  --output_dir results/NTIRE2026_MoCE_IR_AIO3_sigma50 \
  --submission_zip results/NTIRE2026_MoCE_IR_AIO3_sigma50_submission.zip \
  --competition_url "https://www.codabench.org/competitions/12905/#/pages-tab" \
  --dataset_url "https://drive.usercontent.google.com/download?id=1UZA_AEdV5EgqWl9lozYo12YrET-Pno6L&export=download&authuser=0" \
  --runtime_note "MoCE-IR AIO3 denoising inference for NTIRE 2026 image denoising challenge (noise level 50). Results generated on one NVIDIA RTX A6000 GPU."
```

## 6. Output

The generated submission results are not stored in this repository.

Final submission/result package:

```text
https://drive.google.com/file/d/1hXjxCXF-LcQ-gfpSLy_lIMeCm47XLagJ/view?usp=drive_link
```

If you run inference locally, the default output locations are:

- Restored images are written to:

```text
results/NTIRE2026_MoCE_IR_AIO3_sigma50
```

- The submission zip is written to:

```text
results/NTIRE2026_MoCE_IR_AIO3_sigma50_submission.zip
```

The zip contains:

- 200 restored PNG files
- `readme.txt`

All restored images are stored in the root of the zip archive, with the same file names as the input images.

## 7. Runtime Information

Measured runtime for the generated submission:

```text
runtime per image [s] : 2.5034
CPU[1] / GPU[0] : 0
Extra Data [1] / No Extra Data [0] : 1
```

Hardware used for the run:

```text
1 x NVIDIA RTX A6000
```

## 8. Code Files

Main files used for reproduction:

- `src/infer_competition.py`
- `src/net/moce_ir.py`
- `src/utils/test_utils.py`
- `README.md`

## 9. Notes

- This challenge test set does not provide public ground-truth images, so this workflow generates submission results only.
- The repository intentionally excludes large files such as checkpoints, generated results, and checkpoint archives.
- The inference script pads each image to a multiple of 16, runs the model, and crops the restored output back to the original image size.
- Output file names are preserved to satisfy the challenge submission format.
