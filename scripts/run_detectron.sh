# Run Detectron Mask R-CNN on a video stream.
# Usage: bash scripts/run_detectron.sh <input_video_path> <output_path> <CUDA_VISIBLE_DEVICES>
CUDA_VISIBLE_DEVICES=$3 python Detectron.pytorch/tools/infer_video_stream.py --dataset coco --cfg=Detectron.pytorch/configs/baselines/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x_no_aug.yaml --load_detectron=./pretrained/model_large_final.pkl --input_video_path=$1 --output_path=$2
