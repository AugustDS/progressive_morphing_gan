import tensorflow as tf
import numpy as np 

class Generator:

	def __init__(self,
				num_channels        = 3,            # Number of output color channels. Overridden based on dataset.
				resolution          = 128,          # Output resolution. Overridden based on dataset.
				fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
				fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
				fmap_max            = 512,          # Maximum number of feature maps in any layer.
				latent_size         = 512,          # Dimensionality of the latent vectors. None = min(fmap_base, fmap_max).
				normalize_latents   = True,         # Normalize latent vectors before feeding them to the network?
				use_wscale          = True,         # Enable equalized learning rate?
				use_pixelnorm       = True,         # Enable pixelwise feature vector normalization?
				pixelnorm_epsilon   = 1e-8,         # Constant epsilon for pixelwise feature vector normalization.
				use_leakyrelu       = True,         # True = leaky ReLU, False = ReLU.
				dtype               = 'float32',    # Data type to use for activations and outputs.
				fused_scale         = True,        # True = use fused upscale2d + conv2d, False = separate upscale2d layers.
				structure           = 'linear',     # 'linear' = human-readable, 'recursive' = efficient, None = select automatically.
				is_template_graph   = False,        # True = template graph constructed by the Network class, False = actual evaluation.
				is_training         = True,
				is_cpu              = False
				): 
		self.is_cpu              = is_cpu
		self.num_channels        = num_channels           
		self.resolution          = resolution           
		self.fmap_base           = fmap_base
		self.fmap_decay          = fmap_decay
		self.fmap_max            = fmap_max
		self.latent_size         = latent_size
		self.normalize_latents   = normalize_latents
		self.use_wscale          = use_wscale
		self.use_pixelnorm       = use_pixelnorm
		self.pixelnorm_epsilon   = pixelnorm_epsilon
		self.use_leakyrelu       = use_leakyrelu
		self.dtype               = dtype
		self.fused_scale         = fused_scale
		self.structure           = structure
		self.is_template_graph   = is_template_graph
		self.is_training         = is_training 

	def get_outputs_for(self,latents_in):
		from model import lerp, lerp_clip, cset, get_weight, dense, conv2d, apply_bias, \
		leaky_relu, upscale2d, upscale2d_conv2d, downscale2d, conv2d_downscale2d, \
		pixel_norm, minibatch_stddev_layer, G_paper

		return G_paper( latents_in,                  
						self.num_channels,           
						self.resolution,        
						self.fmap_base,          
						self.fmap_decay,          
						self.fmap_max,            
						self.latent_size,
						self.normalize_latents,   
						self.use_wscale,         
						self.use_pixelnorm,       
						self.pixelnorm_epsilon,   
						self.use_leakyrelu,       
						self.dtype,               
						self.fused_scale,         
						self.structure,           
						self.is_template_graph,
						self.is_cpu)  





