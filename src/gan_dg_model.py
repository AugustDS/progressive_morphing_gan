import tensorflow as tf
import numpy as np 
from generator_wrapper import Generator
from discriminator_wrapper import Discriminator

class GAN_DG:
	def __init__(
		self,
		iterator,
		batch_size,
		input_resolution = 128,      # Overwritten depending on configs 
		latent_input_dim = 512,      # Overwritten depending on dataset 
		wgan_lambda      = 5.0,      # Weight for the gradient penalty term.
		wgan_epsilon     = 0.001,    # Weight for the epsilon term, \epsilon_{drift}.
		wgan_target      = 1.0,      # Target value for gradient magnitudes    
		learning_rate    = 0.001,    # Adam learning rate  
		beta1            = 0.0,      # Adam beta 1 
		beta2            = 0.99,     # Adam beta 2
		mirror_augment   = True,
		is_cpu           = False,
		):


		self.iterator = iterator
		self.batch_size = batch_size
		self.input_resolution = input_resolution
		self.latent_input_dim = latent_input_dim
		self.wgan_lambda = wgan_lambda
		self.wgan_epsilon = wgan_epsilon
		self.wgan_target = wgan_target
		self.learning_rate = learning_rate
		self.beta1 = beta1
		self.beta2 = beta2
		self.mirror_augment = mirror_augment

		if is_cpu: 
			fused_scale = False
		else: fused_scale = False
		self.gen = Generator(resolution=self.input_resolution, use_wscale=True, 
			is_cpu=is_cpu, fused_scale = fused_scale, structure = 'linear')
		self.disc = Discriminator(resolution=self.input_resolution, use_wscale=True, 
			is_cpu=is_cpu, fused_scale = fused_scale, structure = 'linear')
		self.latents = tf.random.normal(shape = [batch_size, self.latent_input_dim])
		self.lod_image = tf.cast(tf.get_variable('lod_image', initializer=np.float32(0.0), trainable=False), dtype=tf.float32)

	def lerp_in(self, a, b, t):
		with tf.name_scope('Lerp'):
			return a + (b - a) * t

	# Process images in time (see original progGAN code)
	def process_in_time(self,x):
		if self.mirror_augment:
			s = tf.shape(x)
			mask = tf.random_uniform([s[0], 1, 1, 1], 0.0, 1.0)
			mask = tf.tile(mask, [1, s[1], s[2], s[3]])
			x = tf.where(mask < 0.5, x, tf.reverse(x, axis=[3]))
		s = tf.shape(x)
		y = tf.reshape(x, [-1, s[1], s[2]//2, 2, s[3]//2, 2])
		y = tf.reduce_mean(y, axis=[3, 5], keepdims=True)
		y = tf.tile(y, [1, 1, 1, 2, 1, 2])
		y = tf.reshape(y, [-1, s[1], s[2], s[3]])
		x = self.lerp_in(x, y, self.lod_image - tf.floor(self.lod_image))
		return x


	def get_real_images(self):
		return self.process_in_time(self.iterator.get_next())

	def get_fake_images(self, g_var_scope):
		with tf.variable_scope(g_var_scope, reuse=tf.AUTO_REUSE):
			fake_images = self.gen.get_outputs_for(self.latents)
		return fake_images

	def get_mixed_images(self, real_images, fake_images):
		mixing_factors = tf.random_uniform([self.batch_size, 1, 1, 1], 0.0, 1.0, dtype=fake_images.dtype)
		mixed_images = self.lerp_in(tf.cast(real_images, fake_images.dtype), fake_images, mixing_factors)
		return mixed_images

	def get_logits(self, real_images, fake_images, mixed_images, d_var_scope):

		with tf.variable_scope(d_var_scope, reuse=tf.AUTO_REUSE):
			d_logit_real = self.disc.get_outputs_for(real_images)

		with tf.variable_scope(d_var_scope, reuse=tf.AUTO_REUSE):
			d_logit_fake = self.disc.get_outputs_for(fake_images)

		with tf.variable_scope(d_var_scope, reuse=tf.AUTO_REUSE):
			d_logit_mixed = self.disc.get_outputs_for(mixed_images)

		return d_logit_real, d_logit_fake, d_logit_mixed

	def get_d_loss(self, d_logit_real, d_logit_fake, d_logit_mixed, images_mixed, name_scope):
		with tf.name_scope(name_scope):
			with tf.name_scope('score_loss'):
				d_loss_sc = tf.reduce_mean(d_logit_fake) - tf.reduce_mean(d_logit_real)
			with tf.name_scope('gp_loss'):
				mixed_grads = tf.gradients(tf.reduce_sum(d_logit_mixed),[images_mixed])[0]
				mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1,2,3]))
				gp_loss = tf.reduce_mean(tf.square(mixed_norms - self.wgan_target))*(self.wgan_lambda / (self.wgan_target**2))
			with tf.name_scope('epsilon_penalty'):
				epsilon_penalty = tf.square(tf.reduce_mean(d_logit_real))*self.wgan_epsilon
			d_loss = d_loss_sc + gp_loss + epsilon_penalty

		return d_loss, gp_loss 

	def get_g_loss(self,d_logit_fake,name_scope):
		with tf.name_scope(name_scope):
			g_loss = -tf.reduce_mean(d_logit_fake)
		return g_loss

	def optimizer(self, loss, var_scope, name_scope, lr, be1, be2):
		with tf.name_scope(name_scope):
			tvars = [var for var in tf.trainable_variables() if var_scope in var.name]

			optim = tf.train.AdamOptimizer(learning_rate=lr,beta1=be1, beta2=be2)

			g_and_v = optim.compute_gradients(loss, var_list=tvars)

			train = optim.apply_gradients(g_and_v)

			opt_reset = tf.group([[v.initializer for v in optim.variables()]])

		return train, opt_reset, g_and_v, tvars

	def assign_vars(self, 
		var_scope_d_from = "discriminator", var_scope_d_to="discTMP", 
		var_scope_g_from = "generator", var_scope_to = "genTMP"):
		curr_to_tmp = []
		t_vars = tf.global_variables()
		d_tvars_tmp = [var for var in t_vars if var_scope_to in var.name and 'RMSProp' not in var.name]
		d_tvars_0 = [var for var in t_vars if var_scope_d_from in var.name and 'RMSProp' not in var.name]
		g_tvars_tmp = [var for var in t_vars if var_scope_to in var.name and 'RMSProp' not in var.name]
		g_tvars_0 = [var for var in t_vars if var_scope_g_from in var.name and 'RMSProp' not in var.name]
		for j in range(0, len(d_tvars_tmp)):
			curr_to_tmp.append(d_tvars_tmp[j].assign(d_tvars_0[j]))
		for j in range(0, len(g_tvars_tmp)):
			curr_to_tmp.append(g_tvars_tmp[j].assign(g_tvars_0[j]))
		curr_to_tmp = tf.group(*curr_to_tmp)
		return curr_to_tmp

	def ema_to_weights(self):
		return tf.group(*(tf.assign(var, self.ema.average(var).read_value())
			for var in self.gtvars))

	def save_weight_backups(self):
		return tf.group(*(tf.assign(bck, var.read_value()) for var, bck in zip(self.gtvars, self.backup_vars)))
		
	def restore_weight_backups(self):
		return tf.group(*(tf.assign(var, bck.read_value()) for var, bck in zip(self.gtvars, self.backup_vars)))

	def to_training(self):
		return self.restore_weight_backups()

	def to_testing(self): 
		with tf.control_dependencies([self.save_weight_backups()]):
			return self.ema_to_weights()

	def gan(self):
		real_images  = self.get_real_images()
		fake_images  = self.get_fake_images(g_var_scope = "generator")
		mixed_images = self.get_mixed_images(real_images, fake_images)

		logit_real, logit_fake, logit_mixed = self.get_logits(real_images, 
			fake_images, mixed_images, d_var_scope = "discriminator")

		d_loss, gp_loss = self.get_d_loss(logit_real, logit_fake, 
			logit_mixed, mixed_images, name_scope="d_loss")

		g_loss = self.get_g_loss(logit_fake, name_scope="g_loss")

		d_train, d_opt_reset, d_gv, dtvars = self.optimizer(d_loss, 
			var_scope = "discriminator", name_scope="d_train", lr=self.learning_rate, 
			be1=self.beta1, be2=self.beta2)

		g_train, g_opt_reset, g_gv, self.gtvars = self.optimizer(g_loss, 
			var_scope = "generator", name_scope="g_train", lr=self.learning_rate, 
			be1=self.beta1, be2=self.beta2)

		# Create EMA for snapshots 
		self.ema = tf.train.ExponentialMovingAverage(decay=0.999)
		with tf.control_dependencies([g_train]):
			g_train_op = self.ema.apply(self.gtvars)

		with tf.variable_scope('BackupVariables'):
			self.backup_vars = [tf.get_variable(var.op.name, dtype=var.value().dtype, trainable=False,
				initializer=var.initialized_value()) for var in self.gtvars]

		#------------------------------------------------------------------------------
		# GAN Summaries
		#------------------------------------------------------------------------------
		# Weights and gradients 
		sum_vals = [tf.summary.histogram(var.op.name + "/values", var) for var in dtvars+self.gtvars]
		sum_grads = [tf.summary.histogram(var.op.name + "/gradients", grad) for grad, var in d_gv + g_gv]
		# Scalars 
		gan_summaries = tf.summary.merge(sum_vals+sum_grads+[
		tf.summary.scalar("discriminator_loss", d_loss),
		tf.summary.scalar("generator_loss", g_loss),
		tf.summary.scalar("gp_loss", gp_loss),
		tf.summary.scalar("discriminator_real", tf.reduce_mean(logit_real)),
		tf.summary.scalar("discriminator_fake", tf.reduce_mean(logit_fake))])

		return d_train, d_opt_reset, g_train_op, g_opt_reset, gan_summaries

	def minmax(self):
		real_images    = self.get_real_images()
		fake_images    = self.get_fake_images(g_var_scope = "generator")
		mixed_images   = self.get_mixed_images(real_images, fake_images)

		with tf.variable_scope("worst_calc_d"):
			logit_real_tmp, logit_fake_tmp, logit_mixed_tmp = self.get_logits(
				real_images, fake_images, mixed_images, d_var_scope = "discTMP")

			d_loss_worst, _  = self.get_d_loss(logit_real_tmp, logit_fake_tmp, 
				logit_mixed_tmp, mixed_images, name_scope ="dTMP_loss")

			d_train_worst,_,_,_ = self.optimizer(d_loss_worst, name_scope ="dTMP_train", 
				var_scope = "discTMP", lr=1e-3, be1 = 0.5, be2 = 0.99)

			minmax = - d_loss_worst 

		return d_train_worst, minmax 

	def maxmin(self):
		real_images = self.get_real_images()
		with tf.variable_scope("worst_calc_g"):
			fake_images = self.get_fake_images(g_var_scope="genTMP")

		mixed_images = self.get_mixed_images(real_images, fake_images)
		logit_real_tmp, logit_fake_tmp, logit_mixed_tmp = self.get_logits(
			real_images, fake_images, mixed_images, d_var_scope = "discriminator")

		with tf.variable_scope("worst_calc_g", reuse=tf.AUTO_REUSE):
			g_loss_worst   = self.get_g_loss(logit_fake_tmp, name_scope = "gTMP_loss")
			g_train_worst, _,_,_ = self.optimizer(g_loss_worst, name_scope ="gTMP_train", 
				var_scope = "genTMP", lr=1e-3, be1 = 0.5, be2 = 0.99)

			neg_maxmin, _  = self.get_d_loss(logit_real_tmp, logit_fake_tmp, 
				logit_mixed_tmp, mixed_images, name_scope ="maxmin")

			maxmin = - neg_maxmin

		return g_train_worst, maxmin
