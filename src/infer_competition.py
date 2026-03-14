import argparse
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import lightning.pytorch as pl
from torchvision.transforms import ToTensor

from net.moce_ir import MoCEIR
from utils.test_utils import save_img


class PLTestModel(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.net = MoCEIR(
            dim=opt.dim,
            num_blocks=opt.num_blocks,
            num_dec_blocks=opt.num_dec_blocks,
            levels=len(opt.num_blocks),
            heads=opt.heads,
            num_refinement_blocks=opt.num_refinement_blocks,
            topk=opt.topk,
            num_experts=opt.num_exp_blocks,
            rank=opt.latent_dim,
            with_complexity=opt.with_complexity,
            depth_type=opt.depth_type,
            stage_depth=opt.stage_depth,
            rank_type=opt.rank_type,
            complexity_scale=opt.complexity_scale,
        )

    def forward(self, x):
        return self.net(x)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["MoCE_IR", "MoCE_IR_S"])
    parser.add_argument("--checkpoint_id", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--submission_zip", type=str, default=None)
    parser.add_argument("--readme_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pad_base", type=int, default=16)
    parser.add_argument("--extra_data_flag", type=int, default=1)
    parser.add_argument("--runtime_note", type=str, default="MoCE-IR inference on the NTIRE 2026 denoising test set (sigma=50).")
    parser.add_argument("--competition_url", type=str, default="")
    parser.add_argument("--dataset_url", type=str, default="")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")

    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--num_blocks", nargs="+", type=int, default=[4, 6, 6, 8])
    parser.add_argument("--num_dec_blocks", nargs="+", type=int, default=[2, 4, 4])
    parser.add_argument("--latent_dim", type=int, default=2)
    parser.add_argument("--num_exp_blocks", type=int, default=4)
    parser.add_argument("--num_refinement_blocks", type=int, default=4)
    parser.add_argument("--heads", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--stage_depth", nargs="+", type=int, default=[1, 1, 1])
    parser.add_argument("--with_complexity", action="store_true")
    parser.add_argument("--complexity_scale", type=str, default="max")
    parser.add_argument("--rank_type", type=str, default="spread")
    parser.add_argument("--depth_type", type=str, default="constant")
    parser.add_argument("--topk", type=int, default=1)
    return parser


def configure_model_defaults(args):
    if args.model == "MoCE_IR":
        args.dim = 48 if args.dim is None else args.dim
    elif args.model == "MoCE_IR_S":
        args.dim = 32 if args.dim is None else args.dim
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    return args


def pad_to_base(tensor, base):
    _, _, h, w = tensor.shape
    pad_h = (base - h % base) % base
    pad_w = (base - w % base) % base
    if pad_h == 0 and pad_w == 0:
        return tensor, h, w
    padded = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return padded, h, w


def load_image(image_path):
    image = np.array(Image.open(image_path).convert("RGB"))
    tensor = ToTensor()(image).unsqueeze(0)
    return image, tensor


def resolve_paths(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.readme_path is None:
        args.readme_path = str(output_dir / "readme.txt")
    if args.submission_zip is None:
        args.submission_zip = str(output_dir.parent / f"{output_dir.name}.zip")
    return args


def write_submission_readme(path, runtime_per_image, used_gpu, extra_data_flag, note, competition_url="", dataset_url=""):
    lines = [
        f"runtime per image [s] : {runtime_per_image:.4f}",
        f"CPU[1] / GPU[0] : {0 if used_gpu else 1}",
        f"Extra Data [1] / No Extra Data [0] : {int(extra_data_flag)}",
        f"Other description : {note}",
    ]
    if competition_url:
        lines.append(f"Competition page : {competition_url}")
    if dataset_url:
        lines.append(f"Dataset download link : {dataset_url}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def create_submission_zip(output_dir, readme_path, zip_path):
    output_dir = Path(output_dir)
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for image_path in sorted(output_dir.glob("*.png")):
            zf.write(image_path, arcname=image_path.name)
        zf.write(readme_path, arcname=Path(readme_path).name)


def main():
    args = build_parser().parse_args()
    args = configure_model_defaults(args)
    args = resolve_paths(args)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device.startswith("cuda") and device.type != "cuda":
        raise RuntimeError("CUDA was requested but is not available.")

    ckpt_path = os.path.join(args.ckpt_dir, args.checkpoint_id, "last.ckpt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    input_dir = Path(args.input_dir)
    image_paths = sorted(input_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No PNG images found in: {input_dir}")

    net = PLTestModel.load_from_checkpoint(ckpt_path, opt=args).to(device)
    net.eval()

    output_dir = Path(args.output_dir)
    total_runtime = 0.0

    with torch.inference_mode():
        for image_path in image_paths:
            _, tensor = load_image(image_path)
            tensor = tensor.to(device)
            tensor, orig_h, orig_w = pad_to_base(tensor, args.pad_base)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            restored = net(tensor)
            if isinstance(restored, list) and len(restored) == 2:
                restored = restored[0]
            restored = torch.clamp(restored, 0, 1)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            total_runtime += time.perf_counter() - start

            restored = restored[:, :, :orig_h, :orig_w]
            restored = restored.squeeze(0).permute(1, 2, 0).cpu().numpy()
            restored = np.clip(restored * 255.0, 0, 255).round().astype(np.uint8)
            save_img(str(output_dir / image_path.name), restored)

    runtime_per_image = total_runtime / len(image_paths)
    write_submission_readme(
        args.readme_path,
        runtime_per_image=runtime_per_image,
        used_gpu=device.type == "cuda",
        extra_data_flag=args.extra_data_flag,
        note=args.runtime_note,
        competition_url=args.competition_url,
        dataset_url=args.dataset_url,
    )
    create_submission_zip(output_dir, args.readme_path, args.submission_zip)

    print(f"checkpoint={ckpt_path}")
    print(f"images={len(image_paths)}")
    print(f"runtime_per_image={runtime_per_image:.6f}")
    print(f"output_dir={output_dir}")
    print(f"readme={args.readme_path}")
    print(f"submission_zip={args.submission_zip}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
