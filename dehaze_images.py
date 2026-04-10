"""
AOD-Net Dehazing Inference Script
===================================
Chạy khử sương (image dehazing) trên toàn bộ folder ảnh
bằng mô hình AOD-Net với PONO (PyTorch).

Input  : folder ảnh bị sương mù  (JPEG/PNG/BMP...)
Output : folder ảnh đã khử sương (cùng tên file, định dạng PNG)

Usage:
    python dehaze_images.py \\
        --source   path/to/hazy_images \\
        --output   path/to/dehazed \\
        --weights  weights/AOD_pono_best.pkl \\
        --device   0
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ─── Import model từ thư mục AOD-Net with PONO ────────────────────────────
_PONO_DIR = Path(__file__).parent / "AOD-Net with PONO"
if str(_PONO_DIR) not in sys.path:
    sys.path.insert(0, str(_PONO_DIR))

try:
    from model import AOD_pono_net, AODnet
except ImportError as e:
    raise ImportError(
        f"Không import được model từ '{_PONO_DIR}'. "
        f"Hãy chắc chắn rằng file model.py tồn tại.\nLỗi: {e}"
    )

# ─── Định dạng ảnh hỗ trợ ─────────────────────────────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def get_image_files(directory: Path):
    """Trả về danh sách file ảnh trong directory (đã sort)."""
    return sorted(
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_aodnet(weights_path: str, device: torch.device) -> torch.nn.Module:
    """
    Load AOD-Net PONO từ checkpoint .pkl được lưu bởi pono_train.py.

    Format checkpoint: {'epoch': int, 'state_dict': OrderedDict, 'optimizer': dict}
    State dict keys: conv1.weight/bias ... conv5.weight/bias  (10 keys)
    Lưu ý: PONO dùng affine=False nên không có learnable params → chỉ 10 keys.
    """
    print(f"📦 Load AOD-Net weights: {weights_path}")
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)

    # ── Trích xuất state_dict ─────────────────────────────────────────────
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        # Format chuẩn của pono_train.py
        state_dict = ckpt["state_dict"]
        epoch = ckpt.get("epoch", "?")
        print(f"   → Epoch checkpoint: epoch={epoch}")
    elif isinstance(ckpt, dict) and any(k.startswith("conv") for k in ckpt):
        # Raw state_dict (không có wrapper)
        state_dict = ckpt
        epoch = "?"
    else:
        state_dict = ckpt
        epoch = "?"

    # ── Load vào AOD_pono_net (strict=False vì PONO không có learnable params) ──
    model = AOD_pono_net().to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if unexpected:
        print(f"   ⚠️  Unexpected keys: {unexpected}")
    if missing:
        # PONO (affine=False) và MS không có params → missing là bình thường
        non_pono_missing = [k for k in missing if 'pono' not in k and 'ms' not in k]
        if non_pono_missing:
            print(f"   ⚠️  Missing non-PONO keys: {non_pono_missing}")
            # Fallback về AODnet gốc
            print("   → Thử load AODnet gốc...")
            model = AODnet().to(device)
            model.load_state_dict(state_dict, strict=True)
            print("   → Đã load AODnet (fallback) ✅")
        else:
            print("   → Đã load AOD_pono_net ✅ (PONO params không learnable — bình thường)")
    else:
        print("   → Đã load AOD_pono_net ✅")

    model.eval()
    return model


@torch.no_grad()
def dehaze_folder(opt):
    """Pipeline khử sương toàn bộ folder."""
    device = torch.device(f"cuda:{opt.device}" if opt.device.isdigit() else opt.device)
    print(f"\n{'='*60}")
    print(f"{'AOD-Net Dehazing Inference':^60}")
    print(f"{'='*60}")
    print(f"  Source : {opt.source}")
    print(f"  Output : {opt.output}")
    print(f"  Weights: {opt.weights}")
    print(f"  Device : {device}")
    print(f"{'='*60}\n")

    # ── Load model ────────────────────────────────────────────────────────
    model = load_aodnet(opt.weights, device)

    # ── Chuẩn bị I/O ─────────────────────────────────────────────────────
    source_dir = Path(opt.source)
    output_dir = Path(opt.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory không tồn tại: {source_dir}")

    image_files = get_image_files(source_dir)
    if not image_files:
        print(f"❌ Không tìm thấy ảnh nào trong: {source_dir}")
        return

    print(f"📸 Tìm thấy {len(image_files)} ảnh\n")

    # ── Inference từng ảnh ────────────────────────────────────────────────
    t_start = time.time()
    failed = 0

    for img_path in tqdm(image_files, desc="Dehazing"):
        # Đọc ảnh
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"⚠️  Không đọc được: {img_path}")
            failed += 1
            continue

        h_orig, w_orig = img_bgr.shape[:2]

        # BGR → RGB → tensor [1, 3, H, W] trong [0, 1]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)

        # Forward pass
        dehazed = model(img_tensor)  # output: [1, 3, H, W] in [0, 1]

        # Tensor → NumPy → BGR → lưu ảnh
        dehazed_np = dehazed.squeeze(0).cpu().clamp(0.0, 1.0).numpy()
        dehazed_np = (dehazed_np.transpose(1, 2, 0) * 255.0).astype(np.uint8)
        dehazed_bgr = cv2.cvtColor(dehazed_np, cv2.COLOR_RGB2BGR)

        # Lưu với cùng tên file (extension → .png)
        out_name = img_path.stem + ".png"
        cv2.imwrite(str(output_dir / out_name), dehazed_bgr)

    elapsed = time.time() - t_start
    n_success = len(image_files) - failed

    print(f"\n{'='*60}")
    print(f"{'DEHAZING COMPLETE':^60}")
    print(f"{'='*60}")
    print(f"  Tổng ảnh     : {len(image_files)}")
    print(f"  Thành công   : {n_success}")
    print(f"  Thất bại     : {failed}")
    print(f"  Thời gian    : {elapsed:.2f}s ({n_success/max(elapsed,1e-6):.1f} img/s)")
    print(f"  Output dir   : {output_dir}")
    print(f"{'='*60}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="AOD-Net Dehazing — Khử sương cho toàn bộ folder ảnh",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python dehaze_images.py \\
      --source  target_test/images \\
      --output  dehazed_output \\
      --weights weights/AOD_pono_best.pkl \\
      --device  0
""",
    )
    parser.add_argument("--source",  type=str, required=True,
                        help="Folder chứa ảnh bị sương")
    parser.add_argument("--output",  type=str, default="dehazed_output",
                        help="Folder lưu ảnh đã khử sương (default: dehazed_output)")
    parser.add_argument("--weights", type=str, default="weights/AOD_pono_best.pkl",
                        help="Path đến AOD-Net checkpoint (.pkl / .pt) (default: weights/AOD_pono_best.pkl)")
    parser.add_argument("--device",  type=str, default="0",
                        help="GPU id hoặc 'cpu' (default: 0)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dehaze_folder(args)
