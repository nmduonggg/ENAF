python infer_eunaf.py \
    --template EUNAF_SRResNetxN_1est \
    --testset_tag Flickr2K \
    --N 100 \
    --testset_dir ../../data/Flikr2K/Flickr2K/ \
    --train_stage 1 \
    --n_resblocks 4 \
    --n_estimators 3 \
    --scale 4 \
    --eval_tag psnr \
    --rgb_channel \
    --backbone_name srresnet
    # --visualize