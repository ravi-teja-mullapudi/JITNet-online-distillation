python src/train_jitnet_coco.py \
    --train_data_file=/mnt/disks/tensorflow-disk/coco_train.tfrecords \
    --batch_size=6 \
    --filter_depth_multiplier=1.0 \
    --num_units=2 \
    --train_dir=/tmp/jitnet_coco_pretrained/ \
    --max_number_of_steps=150000 \
    --num_samples_per_epoch=118000 \
    --num_epochs_per_decay=10 \
    --scale=1.0 \
    --height=768 \
    --width=768 \
    --num_clones=1 \
    --optimizer='adam' \
    --learning_rate=0.1
