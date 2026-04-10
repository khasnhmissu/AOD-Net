#!/usr/bin/env bash
# =============================================================================
#  run.sh — Pipeline: Khử sương → Inference YOLO → Đánh giá mAP
#
#  Chạy:  bash run.sh [OPTIONS]
#
#  Các biến có thể override từ môi trường:
#    YOLO_WEIGHTS      : path đến base YOLO weights (default: yolov8n.pt)
#    YOLO_CHECKPOINT   : path đến FusionDA checkpoint (default: weights/best.pt)
#    CONF_THRES        : confidence threshold (default: 0.25)
#    IOU_THRES         : NMS IoU threshold (default: 0.45)
#    DEVICE            : GPU id hoặc 'cpu' (default: 0)
#    SOURCE_DIR        : folder ảnh gốc (có thể bị sương) (default: target_test)
#    LABELS_DIR        : folder ground-truth labels (default: target_test/labels)
#    SKIP_DEHAZE       : set '1' để bỏ qua bước khử sương (default: 0)
# =============================================================================

set -euo pipefail

# ── Màu sắc ───────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'
log()     { echo -e "${GREEN}[RUN]${NC} $*"; }
step()    { echo -e "\n${BOLD}${CYAN}══════════════════════════════════════════════${NC}"; \
            echo -e "${BOLD}${CYAN}  $*${NC}"; \
            echo -e "${BOLD}${CYAN}══════════════════════════════════════════════${NC}"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()     { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Kích hoạt virtual environment ────────────────────────────────────────
VENV_DIR="venv"
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
    log "Đã kích hoạt venv: $(which python)"
else
    warn "Không tìm thấy venv, dùng Python hệ thống."
fi

# ── Cấu hình mặc định (override bằng env vars) ───────────────────────────
YOLO_WEIGHTS="${YOLO_WEIGHTS:-yolov8n.pt}"
YOLO_CHECKPOINT="${YOLO_CHECKPOINT:-weights/best.pt}"
CONF_THRES="${CONF_THRES:-0.25}"
IOU_THRES="${IOU_THRES:-0.45}"
DEVICE="${DEVICE:-0}"
SOURCE_DIR="${SOURCE_DIR:-target_test/target_test/val}"
LABELS_DIR="${LABELS_DIR:-${SOURCE_DIR}/labels}"
SKIP_DEHAZE="${SKIP_DEHAZE:-0}"
AOD_WEIGHTS="${AOD_WEIGHTS:-weights/AOD_pono_best.pkl}"

DEHAZED_DIR="dehazed_output"          # Folder ảnh sau khử sương
PREDICTS_DIR="predicts"               # Folder predictions YOLO
RESULTS_DIR="results"                 # Folder lưu kết quả đánh giá
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# ── Tạo output dirs ───────────────────────────────────────────────────────
mkdir -p "$DEHAZED_DIR" "$PREDICTS_DIR" "$RESULTS_DIR"

# ── In cấu hình ───────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          AOD-Net + YOLO Detection Pipeline                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "  Source images    : $SOURCE_DIR"
echo "  Labels dir       : $LABELS_DIR"
echo "  AOD-Net weights  : $AOD_WEIGHTS"
echo "  YOLO weights     : $YOLO_WEIGHTS"
echo "  YOLO checkpoint  : $YOLO_CHECKPOINT"
echo "  Conf threshold   : $CONF_THRES"
echo "  IoU threshold    : $IOU_THRES"
echo "  Device           : $DEVICE"
echo "  Skip dehaze      : $SKIP_DEHAZE"
echo "  Timestamp        : $TIMESTAMP"
echo ""

# ══════════════════════════════════════════════════════════════════════════
# BƯỚC 1: KHỬ SƯƠNG (AOD-Net)
# ══════════════════════════════════════════════════════════════════════════
if [ "$SKIP_DEHAZE" = "1" ]; then
    warn "Bỏ qua bước khử sương (SKIP_DEHAZE=1)"
    INFERENCE_SOURCE="$SOURCE_DIR"
else
    step "BƯỚC 1/3 — Khử sương bằng AOD-Net"

    # Tìm folder chứa ảnh trong SOURCE_DIR
    # Hỗ trợ cấu trúc: target_test/images/ hoặc target_test/*.jpg
    if [ -d "${SOURCE_DIR}/images" ]; then
        HAZE_IMG_DIR="${SOURCE_DIR}/images"
    else
        HAZE_IMG_DIR="${SOURCE_DIR}"
    fi

    log "Nguồn ảnh sương: $HAZE_IMG_DIR"
    log "Output khử sương: $DEHAZED_DIR"

    python dehaze_images.py \
        --source    "$HAZE_IMG_DIR" \
        --output    "$DEHAZED_DIR" \
        --weights   "$AOD_WEIGHTS" \
        --device    "$DEVICE"

    INFERENCE_SOURCE="$DEHAZED_DIR"
    log "✅ Khử sương hoàn tất → $DEHAZED_DIR"
fi

# ══════════════════════════════════════════════════════════════════════════
# BƯỚC 2: INFERENCE YOLO
# ══════════════════════════════════════════════════════════════════════════
step "BƯỚC 2/3 — Chạy YOLO Detection"

# Kiểm tra checkpoint
[ ! -f "$YOLO_CHECKPOINT" ] && err "Không tìm thấy YOLO checkpoint: $YOLO_CHECKPOINT"

log "Source: $INFERENCE_SOURCE"
log "Output: $PREDICTS_DIR"

python inference_base.py \
    --weights     "$YOLO_WEIGHTS" \
    --checkpoint  "$YOLO_CHECKPOINT" \
    --source      "$INFERENCE_SOURCE" \
    --output      "$PREDICTS_DIR" \
    --conf-thres  "$CONF_THRES" \
    --iou-thres   "$IOU_THRES" \
    --device      "$DEVICE"

log "✅ Inference hoàn tất → $PREDICTS_DIR"

# ══════════════════════════════════════════════════════════════════════════
# BƯỚC 3: ĐÁNH GIÁ mAP
# ══════════════════════════════════════════════════════════════════════════
step "BƯỚC 3/3 — Đánh giá mAP"

# Kiểm tra labels
[ ! -d "$LABELS_DIR" ] && err "Không tìm thấy labels dir: $LABELS_DIR"

REPORT_FILE="${RESULTS_DIR}/eval_${TIMESTAMP}.txt"
PR_CURVE_FILE="${RESULTS_DIR}/pr_curves_${TIMESTAMP}.png"

log "Labels   : $LABELS_DIR"
log "Predicts : $PREDICTS_DIR"
log "Report   : $REPORT_FILE"

python map_evaluation.py \
    --labels    "$LABELS_DIR" \
    --predicts  "$PREDICTS_DIR" \
    --conf      0.5 \
    --save-plot "$PR_CURVE_FILE" \
    2>&1 | tee "$REPORT_FILE"

# ══════════════════════════════════════════════════════════════════════════
# TÓM TẮT
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ✅  PIPELINE HOÀN TẤT                                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "  Ảnh khử sương  : $DEHAZED_DIR/"
echo "  Predictions     : $PREDICTS_DIR/"
echo "  Báo cáo mAP     : $REPORT_FILE"
echo "  PR Curve        : $PR_CURVE_FILE"
echo ""
