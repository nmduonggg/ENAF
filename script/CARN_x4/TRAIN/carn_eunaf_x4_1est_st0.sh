python train_eunaf.py \
    --template EUNAF_CARNxN_1est \
    --N 14 \
    --scale 4 \
    --train_stage 0 \
    --max_epochs 30 \
    --lr 0.001 \
    --testset_tag='Set14RGB' \
    --testset_dir='../../data/' \
    --n_estimators 3 \
    --trainset_preload 200 \
    --rgb_channel \
    # --wandb \
    # --lr 0.00
    # --max_load 1000