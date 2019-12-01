# Adversarail camera stickers
A very limted implementation of [Adversarial camera stickers: A physical camera-based attack on deep learning systems](https://arxiv.org/abs/1904.00759).

<p align="center">
  <img width="300" height="300" src="/image/1200px-STOP_sign.jpg">
</p>

Before the attack:
```
class: harvester, idx: 595, logit: 6.0803
class: boathouse, idx: 449, logit: 6.5917
class: scoreboard, idx: 781, logit: 6.7801
class: chainlink_fence, idx: 489, logit: 7.7015
class: birdhouse, idx: 448, logit: 7.9526
class: mailbox, idx: 637, logit: 7.9899
class: barn, idx: 425, logit: 7.9944
class: traffic_light, idx: 920, logit: 8.5474
class: water_tower, idx: 900, logit: 9.2191
class: street_sign, idx: 919, logit: 13.6607
```

<p align="center">
  <img width="300" height="300" src="https://i.imgur.com/Y3V1UsE.png">
</p>

After the attack:
```
class: radio_telescope, idx: 755, logit: 7.8390
class: lumbermill, idx: 634, logit: 7.9665
class: barrow, idx: 428, logit: 8.0820
class: mailbox, idx: 637, logit: 8.2715
class: thresher, idx: 856, logit: 8.9653
class: garbage_truck, idx: 569, logit: 9.0486
class: birdhouse, idx: 448, logit: 9.3717
class: tractor, idx: 866, logit: 9.3863
class: lawn_mower, idx: 621, logit: 9.6112
class: harvester, idx: 595, logit: 14.0861
```

## Differences from the original paper
This repository provides:
- only digital attacks (physical attacking case is not checked)
- one image attack  
  The original paper explores color dots by using many images of the true label.
- very simplified implementations
  - The number of dots is fixed to 9 (no special reason).
  - Positions and colors of the dots are directly optimized by a simple gradient descent.
  - No constraint of physically realizable perturbation.
  - Color transformation (defined in the eq.9 in the original paper) is not used.
  - Various parameters such as effective radius are not well explored.


## Environment
Even though VSCode Remove Develepment environment is assumed, the script essentially depends on `PyTorch`, `torchvision`, `NumPy` and `skimage`.


## Running script
This repository includes an image which will be attacked [here](https://github.com/yoheikikuta/adversarial-camera-stickers/tree/master/image); it's a street_sign image (ImageNet-1000 class label idx is `919`).

- Targeted attack  
  You can choose a target label which follows ImageNet-1000 class labels (default: `595` (harvester)).
```
$ python main.py --is_targeted (--target_label_idx 595)
```

- Non-targeted attack  
  This is just to reduce a NEGATIVE loss function of the true label.
```
$ python main.py
```

`./notebook/experiment.ipynb` includes the same experiment with some visualizations.
