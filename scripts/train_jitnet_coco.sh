# Train JITNet on the COCO dataset.
# Usage: bash scripts/train_jitnet_coco.sh <train_data_file> <checkpoint_dir>
python src/train_jitnet_coco.py \
    --train_data_file=$1 \
    --batch_size=6 \
    --filter_depth_multiplier=1.0 \
    --num_units=2 \
    --train_dir=$2 \
    --max_number_of_steps=150000 \
    --num_samples_per_epoch=118000 \
    --num_epochs_per_decay=10 \
    --scale=1.0 \
    --height=768 \
    --width=768 \
    --num_clones=1 \
    --optimizer='adam' \
    --learning_rate=0.1
