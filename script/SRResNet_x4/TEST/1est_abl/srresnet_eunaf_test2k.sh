python test_eunaf_by_patches_1est_abl.py \
    --template EUNAF_SRResNetxN_1est \
    --testset_tag Test2K \
    --N 100 \
    --testset_dir ../../data/test2k/ \
    --train_stage 1 \
    --n_resblocks 16 \
    --n_estimators 3 \
    --scale 4 \
    --eval_tag psnr \
    --rgb_channel \
    --backbone_name srresnet
    # --visualize