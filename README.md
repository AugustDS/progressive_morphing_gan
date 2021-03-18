### Progressive Growing of GANs with Network Morphism

This repository contains the code to progressively grow a GAN with network morphism. We used the low-capacity network structure from Appendix A.2 of the [originally proposed progressive GAN model](https://arxiv.org/abs/1710.10196) and combined it with methods from [network morphism](http://proceedings.mlr.press/v48/wei16.pdf) to speed up an stablize training, while keeping track of the training stability by analysing the [duality gap](https://papers.nips.cc/paper/9377-a-domain-agnostic-measure-for-monitoring-and-evaluating-gans.pdf). We developed the idea after an extensive experimental investigation of progressive growth for GANs during a project at the [data analytics lab](http://www.da.inf.ethz.ch) at ETH Zürich. Please see the [project presentation](https://drive.google.com/file/d/1rWrr843RlurrtVGk5q_Yn1Ek_xv6ZMBu/view?usp=sharing) for more details.

During progressive growth the resolution of the generated images is doubled by smoothly interpolating between consecutive convolutional blocks with up- and downsampling layers inbetween. To ensure training stability, each _interpolation_ (or _transition_) phase lasts until the discriminator has seen 600T real images, the same duration as during the subsequent _stabilization_ phase. To speed up the transition without harming stability, we propose to use network morphism to grow the new convolution blocks from the previous network layers. Because the up- and downsampling and convolution operations are linear the morphed output remains the same as long as the dynamic weight scaling is reversed. Once we have grown the new block, we interpolate from an identity mapping after each convolution towards the non-linear operations (activation function and pixel-wise feature vector normalization). This interpolation phase only lasts for ~20T real images resulting in a significant training speed-up without harming stability or performance. 

<img src="/teaser.png"  width="435" height="220">

**Teaser Figure:** Network morphism approach and generated images
### Important 

The [morphing operation code](http://proceedings.mlr.press/v48/wei16.pdf) cannot be publicly shared, so for any questions please send me a mail [augustschdmnt@gmail.com](mailto:augustschdmnt@gmail.com). 


### Steps

1. Download the [celeb-a](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) (218x178 pixels) or [lsun](https://www.yf.io/p/lsun) (256x360 pixels) dataset, split the images (.jpeg's) in training (80%), validation (10%) and testing (10%) folders, for computing the duality gap. 
2. Edit the `model.ini` file in the `config` directory. Particularly, the paths to `train/val/test_dir` and the `checkpoint_dir`. 
3. Run the training script: `> python3 run_training_dg.py`
4. Analyse training progress and snapshots with tensorboard (tf.event files are saved in the checkpoints folder)

### License 

MIT License 
