python train_eunaf_no_freeze_srr_1est.py \
    --template EUNAF_SRResNetxN_1est \
    --N 14 \
    --scale 4 \
    --train_stage 1 \
    --max_epochs 30 \
    --lr 0.00005 \
    --testset_tag='Set14RGB' \
    --testset_dir='../../data/' \
    --n_estimators 3 \
    --trainset_preload 200 \
    --rgb_channel \
    --weight '/mnt/disk1/nmduong/FusionNet/Supernet-SR/src/checkpoints/EUNAF_EDSRx2_bl_x2_nb4_nf64_st0/_best.t7' \
    # --wandb \