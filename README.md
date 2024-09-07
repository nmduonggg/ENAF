## ENAF Sample Implementation

This is the sample source code provided for WACV 2025 submission: "ENAF: A Multi-Exit Network with an Adaptive Patch Fusion
for Large Image Super Resolution". 

### Pretrained weights:
- Weights are stored in [weights](./weights)

### Adjust configuration
- Adjust configuration and weight path in folder [template](./template/)
- Importantly, notice the test data dir path.

### Inference
- Run the script placed in each network's script:
```
$ script/FSRCNN_x4/TEST/1est/fsrcnn_eunaf_test8k.sh  # FSRCNN
$ script/CARN_x4/TEST/1est/carn_eunaf_test8k.sh  # CARN
$ script/SRResNet_x4/TEST/1est/srresnet_eunaf_test8k.sh  # SRResNet
```
