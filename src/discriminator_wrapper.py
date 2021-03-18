import tensorflow as tf
import numpy as np 

class Discriminator:

	def __init__(self,
				num_channels        = 3,            # Number of output color channels. Overridden based on dataset.
				resolution          = 128,          # Output resolution. Overridden based on dataset.
				fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
				fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
				fmap_max            = 512,          # Maximum number of feature maps in any layer.
				use_wscale          = True,         # Enable equalized learning rate?
				mbstd_group_size    = 4,            # Group size for the minibatch standard deviation layer, 0 = disable.
				dtype               = 'float32',    # Data type to use for activations and outputs.
				fused_scale         = True,        # True = use fused upscale2d + conv2d, False = separate upscale2d layers.
				structure           = 'linear',     # 'linear' = human-readable, 'recursive' = efficient, None = select automatically.
				is_template_graph   = False,        # True = template graph constructed by the Network class, False = actual evaluation.
				use_loss_scaling    = False,        # Dynamic Loss Scaling
				minibatch_size      = 1,
				is_cpu              = False
				): 

		self.is_cpu              = is_cpu
		self.num_channels        = num_channels           
		self.resolution          = resolution           
		self.fmap_base           = fmap_base
		self.fmap_decay          = fmap_decay
		self.fmap_max            = fmap_max
		self.use_wscale          = use_wscale
		self.mbstd_group_size    = mbstd_group_size
		self.dtype               = dtype
		self.fused_scale         = fused_scale
		self.structure           = structure
		self.is_template_graph   = is_template_graph
		self.use_loss_scaling    = use_loss_scaling
		self.minibatch_size      = minibatch_size


	def get_outputs_for(self,images_in):
		from model import lerp, lerp_clip, cset, get_weight, dense, conv2d,apply_bias,\
		leaky_relu, upscale2d, upscale2d_conv2d, downscale2d, conv2d_downscale2d, \
		pixel_norm, minibatch_stddev_layer, D_paper

		return D_paper(images_in,
						self.num_channels,           
						self.resolution,        
						self.fmap_base,          
						self.fmap_decay,          
						self.fmap_max,            
						self.use_wscale,         
						self.mbstd_group_size,       
						self.dtype,               
						self.fused_scale,         
						self.structure,           
						self.is_template_graph,
						self.is_cpu)
