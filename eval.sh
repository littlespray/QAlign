# python q_align/evaluate/hy_wx_eval.py --model-path T2VQA-full-ft-results-epoch2 --device cuda:0
DUAL_IMAGE_MODE=True

EXTRA_FLAGS=""
if [ "$DUAL_IMAGE_MODE" = "True" ] || [ "$DUAL_IMAGE_MODE" = "true" ] || [ "$DUAL_IMAGE_MODE" = "1" ]; then
  EXTRA_FLAGS="--dual-image-mode"
fi

python q_align/evaluate/eval_ie.py --model-path outputs/dual-epoch60-b128-2 --device cuda:0 ${EXTRA_FLAGS}