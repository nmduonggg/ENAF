python create_hmap.py \
    --template EUNAF_SRResNetxN_1est \
    --N 14 \
    --scale 4 \
    --train_stage 0 \
    --n_estimators 3 \
    --testset_tag LQGT\
    --testset_dir ../../data/DIV2K/TMP/DIV2K_valid_HR_sub/\
    --rgb_channel \
    --weight '/mnt/disk1/nmduong/FusionNet/Supernet-SR/src/checkpoints/EUNAF_EDSRx2_bl_x2_nb4_nf64_st0/_best.t7' \
    # --wandb \``