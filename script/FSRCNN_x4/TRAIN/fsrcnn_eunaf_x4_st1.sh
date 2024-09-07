python train_eunaf.py \
    --template EUNAF_FSRCNNxN_1est \
    --N 14 \
    --scale 4 \
    --train_stage 1 \
    --max_epochs 30 \
    --lr 0.0002 \
    --testset_tag='Set14RGB' \
    --testset_dir='../../data/' \
    --n_estimators 3 \
    --trainset_preload 200 \
    --rgb_channel \
    # --wandb \