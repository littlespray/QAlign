# python q_align/evaluate/hy_wx_eval.py --model-path T2VQA-full-ft-results-epoch2 --device cuda:0
DUAL_IMAGE_MODE=True
python q_align/evaluate/eval_ie.py --model-path outputs/base-epoch8-b32 --device cuda:0 --dual-image-mode ${DUAL_IMAGE_MODE}