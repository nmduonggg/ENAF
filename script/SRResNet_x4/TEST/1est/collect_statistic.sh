python collect_psnr_unc.py \
    --template EUNAF_SRResNetxN_1est \
    --testset_tag LQGT \
    --N 100 \
    --testset_dir ../../data/DIV2K/TMP/DIV2K_valid_HR_sub/ \
    --train_stage 1 \
    --n_resblocks 16 \
    --n_estimators 3 \
    --scale 4 \
    --eval_tag ssim \
    --rgb_channel \
    --analyze_dir experiments/ANALYZE\

    # --visualize