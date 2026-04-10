#!/usr/bin/env bash
# =============================================================================
#  setup.sh — Cài đặt môi trường đầy đủ cho AOD-Net + FusionDA Inference
#
#  Chạy:  bash setup.sh
#  Yêu cầu: Python ≥ 3.8, CUDA driver đã cài (nếu dùng GPU)
# =============================================================================

set -euo pipefail

# ── Màu sắc terminal ──────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'
log()  { echo -e "${GREEN}[SETUP]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN] ${NC} $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── 0. Kiểm tra Python ────────────────────────────────────────────────────
log "Kiểm tra Python..."
PYTHON=$(command -v python3 || command -v python || true)
[ -z "$PYTHON" ] && err "Không tìm thấy Python 3. Hãy cài Python >= 3.8."
PY_VER=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
log "  → Python $PY_VER tại $PYTHON"

# ── 1. Tạo virtual environment ────────────────────────────────────────────
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
    log "Tạo virtual environment tại ./${VENV_DIR}..."
    $PYTHON -m venv "$VENV_DIR"
else
    warn "Virtual environment đã tồn tại, bỏ qua tạo mới."
fi

source "$VENV_DIR/bin/activate"
log "  → Đã kích hoạt venv: $(which python)"

# ── 2. Nâng cấp pip / setuptools ─────────────────────────────────────────
log "Nâng cấp pip & setuptools..."
pip install --upgrade pip setuptools wheel -q

# ── 3. Cài đặt PyTorch (CUDA 11.8) ───────────────────────────────────────
log "Cài đặt PyTorch (CUDA 11.8)..."
# Nếu server dùng CUDA phiên bản khác, sửa index-url tương ứng:
#   CUDA 12.1: https://download.pytorch.org/whl/cu121
#   CPU only : https://download.pytorch.org/whl/cpu
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 -q

# ── 4. Cài đặt Ultralytics (YOLOv8 / YOLO26) ─────────────────────────────
log "Cài đặt Ultralytics..."
pip install ultralytics -q

# ── 5. Cài đặt các dependencies khác ─────────────────────────────────────
log "Cài đặt dependencies còn lại..."
pip install \
    opencv-python-headless \
    numpy \
    tqdm \
    pyyaml \
    gdown \
    matplotlib \
    -q

# ── 6. Tải dataset target_test ────────────────────────────────────────────
log "Tải dataset target_test từ Google Drive..."
if [ ! -d "target_test" ]; then
    gdown 1TRnmVqgujZucqvwLDCvKA55z_j3usUtm -O target_test.zip
    log "Giải nén target_test.zip..."
    unzip -q target_test.zip -d target_test
    rm -f target_test.zip
    log "  → Dataset đã giải nén vào ./target_test/"
else
    warn "Thư mục target_test đã tồn tại, bỏ qua tải về."
fi

# ── 7. Kiểm tra cấu trúc dataset ─────────────────────────────────────────
log "Kiểm tra cấu trúc dataset..."
if [ -d "target_test" ]; then
    echo "  Nội dung target_test/:"
    ls -la target_test/ | head -20
fi

# ── 8. Giải nén AOD-Net PONO weights từ ponomodels.zip ─────────────────────
log "Thiết lập thư mục weights..."
mkdir -p weights

PONO_WEIGHTS="weights/AOD_pono_best.pkl"
PONOMODELS_ZIP="AOD-Net with PONO/ponomodels.zip"

if [ ! -f "$PONO_WEIGHTS" ]; then
    if [ -f "$PONOMODELS_ZIP" ]; then
        log "Giải nén AOD-Net PONO weights từ ${PONOMODELS_ZIP}..."
        python3 - <<'PYEOF'
import zipfile, os

zip_path = "AOD-Net with PONO/ponomodels.zip"
out_path = "weights/AOD_pono_best.pkl"
src_name = "ponomodels/aod-xavier/AOD_9.pkl"  # epoch 9 = best

with zipfile.ZipFile(zip_path) as z:
    data = z.read(src_name)
with open(out_path, "wb") as f:
    f.write(data)
print(f"  → Saved: {out_path} ({len(data):,} bytes)")
PYEOF
        log "  → AOD-Net PONO weights đã được giải nén ✅"
    else
        warn "Không tìm thấy ${PONOMODELS_ZIP}."
        warn "→ Hãy đặt file AOD_pono_best.pkl vào thư mục weights/ thủ công."
    fi
else
    warn "AOD-Net weights đã tồn tại tại ${PONO_WEIGHTS}, bỏ qua."
fi

# ── 9. Tóm tắt ────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "✅  SETUP HOÀN TẤT"
echo "═══════════════════════════════════════════════════════════════"
echo "  Môi trường : ./${VENV_DIR}/"
echo "  Dataset     : ./target_test/"
echo "  Weights dir : ./weights/"
echo ""
echo "  Bước tiếp theo:"
echo "    1. Đặt YOLO checkpoint vào ./weights/  (e.g. best.pt)"
echo "    2. Kiểm tra AOD-Net weights tại ./weights/AOD_pono_best.pkl"
echo "    3. Chạy:  bash run.sh"
echo "═══════════════════════════════════════════════════════════════"
