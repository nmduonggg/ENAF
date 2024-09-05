python train_eunaf_no_freeze_srr.py \
    --template EUNAF_SRResNetxN \
    --N 14 \
    --scale 4 \
    --train_stage 2 \
    --max_epochs 300 \
    --lr 0.001 \
    --testset_tag='Set14RGB' \
    --testset_dir='../../data/' \
    --trainset_preload 100 \
    --n_estimators 4 \
    --rgb_channel \
    --weight './checkpoints/EUNAF_SRResNetxN_x2_nb16_nf64_st1/_best.t7' \
    # --lr 0.00
    # --max_load 1000