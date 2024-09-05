python infer_eunaf_by_patches_1est.py \
    --template EUNAF_CARNxN_1est \
    --testset_tag Test8K \
    --N 100 \
    --testset_dir ../../data/test8k/ \
    --train_stage 1 \
    --n_resblocks 4 \
    --n_estimators 3 \
    --scale 4 \
    --eval_tag psnr \
    --rgb_channel \
    --backbone_name carn
    # --visualize