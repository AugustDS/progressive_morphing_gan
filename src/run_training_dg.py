
import logging
import os
import shutil
from datetime import datetime
import numpy as np
import tensorflow as tf
import collections 
from config_loader import Config
from schedule import TrainingSchedule
from dataset import Dataset
from gan_dg_model import GAN_DG
from model_utils import restore_model, construct_model_name
import time
import numpy as np
import sys
from netmorph_utils import pad_zeros, atleast_4d
from deconv_linalg import conv_filters_full, \
    deconv_filters_full, deconv_filters_full_type2
from netmorph_depth import normalize_std, get_loss_mse, decomp_filters_lsq_iter


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('config','../config/model.ini', 'input configuration')
tf.flags.DEFINE_string('section', 'DEFAULT', 'input configuration section')


def train():
    # Load the config variables
    cfg = Config(FLAGS.config)
    steps_per_phase    = cfg.get('steps_per_phase')
    add_steps_exp      = cfg.get('add_steps_exp')
    min_steps          = cfg.get('min_steps')
    dg_steps           = cfg.get('dg_steps')
    checkpoints_folder = cfg.get('checkpoints_folder')
    path_input_train   = cfg.get('path_input_train')
    path_input_val     = cfg.get('path_input_test')
    path_input_test    = cfg.get('path_input_val')
    input_resolution   = cfg.get('input_resolution')
    batch_size         = cfg.get('batch_size')
    learning_rate      = cfg.get('learning_rate')
    latent_input_dim   = cfg.get('latent_input_dim') 
    model_name         = cfg.get('model_name')
    seed               = cfg.get('seed')
    n_critic           = cfg.get('n_critic')
    version            = cfg.get('version')
    inference_steps    = cfg.get('inference_steps')
    is_training        = cfg.get('is_training')
    dg_freq            = cfg.get('dg_freq')
    sum_freq           = cfg.get('sum_freq')
    snap_freq          = cfg.get('snap_freq')
    is_cpu             = cfg.get('is_cpu')
    save_model_freq    = cfg.get('save_model_freq')
    is_testing         = cfg.get('is_testing')
    dataset_name       = cfg.get("dataset_name")
    growth_ip_steps    = cfg.get("act_ip_steps")

    # Define training steps and images per phase
    kimg_per_phase = steps_per_phase*batch_size*n_critic/1000.0
    num_phases = 2*(int(np.log2(input_resolution))-2)+1
    training_steps = int(num_phases*steps_per_phase) + 1 + add_steps_exp 
    if training_steps < min_steps:
        training_steps = min_steps


    # When training is continued model_name is non-empty
    load_model = model_name is not None and model_name != ""
    if not load_model:
        model_basename = cfg.get('model_basename')
        model_name = construct_model_name(model_basename, version)

    # Define checkpoint folder 
    checkpoint_folder = os.path.join(checkpoints_folder, model_name)
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    # For traceability of experiments, copy the config file
    shutil.copyfile(FLAGS.config, os.path.join(checkpoint_folder,
                                               "{}__{}.ini".format(
                                                   model_name,
                                                   FLAGS.section)))

    # One training step
    #---------------------------------------------------------------
    def training_step(sess,step,model,n_critic,d_train,
        sum_freq,g_train,gan_summary,snap_freq,snap_summary,
        dg_freq,current_to_tmp_op,iter_val_init_op,dg_steps,
        d_train_worst,iter_test_init_op,minmax,g_train_worst,
        maxmin,iter_train_init_op):

        # Train discriminator for n_critic
        for critic_iter in range(n_critic):
            results = sess.run(d_train)

        # Train generator and add summaries every sum_freq steps
        if step%sum_freq == 0 and step>0:
            results, summary1 = sess.run([g_train, gan_summary])
            train_writer.add_summary(summary1, step)
            train_writer.flush()
        else: 
            results = sess.run(g_train)

        # Chane to EMA variables when adding snapshot summary 
        # (for inference we take exp. running avg. of weights)
        if step%snap_freq == 0 and step>0:
            sess.run(model.to_testing())
            summary2 = sess.run(snap_summary)
            train_writer.add_summary(summary2, step)
            train_writer.flush()
            sess.run(model.to_training())

        duality_gap = None
        maxmin_value = None
        minmax_value = None

        # Compute duality gap at dg_freq
        if step%dg_freq==0 and step>0:

            # Assign current weights to g_worst, d_worst 
            sess.run(current_to_tmp_op)

            # Train d_worst on validation data
            sess.run(iter_val_init_op)
            for i in range(0,dg_steps):
                _ = sess.run(d_train_worst)

            # Compute minmax on test data
            sess.run(iter_test_init_op)
            minmax_value = sess.run(minmax)

            # Train g_worst on validation data
            sess.run(iter_val_init_op)
            for j in range(0,dg_steps):
                _ = sess.run(g_train_worst)

            # Compute maxmin on test data
            sess.run(iter_test_init_op)
            maxmin_value = sess.run(maxmin)

            # Compute DG
            duality_gap = minmax_value - maxmin_value
            
            # Add to summaries 
            summary3 = tf.Summary()
            summary3.value.add(tag="minmax", simple_value=minmax_value)
            summary3.value.add(tag="maxmin", simple_value=maxmin_value)
            summary3.value.add(tag="duality_gap", simple_value=duality_gap)
            train_writer.add_summary(summary3, step)
            train_writer.flush()
            
            # Print in log file 
            logging.info('-----------Step %d:-------------' % step)
            logging.info('  Time: {}'.format(
            datetime.now().strftime('%b-%d-%I%M%p-%G')))
            logging.info('  Duality Gap  : {}'.format(duality_gap))
            
            # Reinit train OP's 
            sess.run(iter_train_init_op)
        return duality_gap, maxmin_value, minmax_value 

    # To undo dynamic weight scaling at transition 
    #---------------------------------------------------------------
    def get_wscale(shape, gain=np.sqrt(2),fan_in=None):
        if fan_in is None: fan_in = np.prod(shape[:-1])
        std = gain / np.sqrt(fan_in) # He init
        return std 


    # Built graph
    #---------------------------------------------------------------
    graph = tf.Graph()
    with graph.as_default():

        # Get datasets, iterators and initializers (train/val/test)
        #---------------------------------------------------------------
        dataset_train = Dataset(path_input_train, batch_size=batch_size, new_size=input_resolution, what_dataset=dataset_name).get_dataset()
        dataset_val   = Dataset(path_input_val, batch_size=batch_size, new_size=input_resolution, what_dataset=dataset_name).get_dataset()
        dataset_test  = Dataset(path_input_test, batch_size=batch_size, new_size=input_resolution, what_dataset=dataset_name).get_dataset()
        data_iterator = tf.data.Iterator.from_structure(dataset_train.output_types, dataset_train.output_shapes)
        iter_train_init_op = data_iterator.make_initializer(dataset_train)
        iter_test_init_op = data_iterator.make_initializer(dataset_val)
        iter_val_init_op = data_iterator.make_initializer(dataset_test)

        # Create model
        #---------------------------------------------------------------
        model = GAN_DG(
            iterator = data_iterator,
            batch_size = batch_size,
            input_resolution = input_resolution,
            latent_input_dim = latent_input_dim,
            learning_rate = learning_rate,
            is_cpu = is_cpu)

        # Get training OPs
        #---------------------------------------------------------------
        d_train, d_opt_reset, g_train, g_opt_reset, g_summary  = model.gan()
        current_to_tmp_op = model.assign_vars()
        d_train_worst, minmax = model.minmax()
        g_train_worst, maxmin = model.maxmin()

        # Check snapshots during morphing
        #---------------------------------------------------------------
        im_fk = model.get_fake_images(g_var_scope = "generator")
        snap_summary = tf.summary.image("snapshot_fake", 
            tf.transpose(im_fk, [0,2,3,1]), max_outputs = 2)
        snap_summary_1 = tf.summary.image("snapshot_after_morph",
            tf.transpose(im_fk, [0,2,3,1]), max_outputs = 4)
        snap_summary_2 = tf.summary.image("snapshot_before_morph",
            tf.transpose(im_fk, [0,2,3,1]), max_outputs = 4)


        # Create lod placeholder, assign lod values (lod_in_g = lod_in_d), add to summary 
        #---------------------------------------------------------------
        lod_in_g = tf.placeholder(tf.float32, name='lod_in_g', shape=[])
        lod_in_d = tf.placeholder(tf.float32, name='lod_in_d', shape=[])
        lod_in_im = tf.placeholder(tf.float32, name='lod_in_im', shape=[])
        lod_in_grow = tf.placeholder(tf.float32, name='lod_in_grow', shape=[])

        lod_image_a = [v for v in tf.global_variables() if v.name == "lod_image:0"][0]
        lod_g_a = [v for v in tf.global_variables() if v.name == "generator/lod_g:0"][0]
        lod_d_a = [v for v in tf.global_variables() if v.name == "discriminator/lod_d:0"][0]
        lod_gtmp_a = [v for v in tf.global_variables() if v.name == "worst_calc_g/genTMP/lod_g:0"][0]
        lod_dtmp_a = [v for v in tf.global_variables() if v.name == "worst_calc_d/discTMP/lod_d:0"][0]

        lod_grow_g_a = [v for v in tf.global_variables() if v.name == "generator/lod_grow:0"][0]
        lod_grow_d_a = [v for v in tf.global_variables() if v.name == "discriminator/lod_grow:0"][0]
        lod_grow_gtmp_a = [v for v in tf.global_variables() if v.name == "worst_calc_g/genTMP/lod_grow:0"][0]
        lod_grow_dtmp_a = [v for v in tf.global_variables() if v.name == "worst_calc_d/discTMP/lod_grow:0"][0]

        lod_assign_ops = [tf.assign(lod_g_a, lod_in_g), tf.assign(lod_d_a, lod_in_d),
                          tf.assign(lod_gtmp_a, lod_in_g), tf.assign(lod_dtmp_a, lod_in_d)]

        lod_grow_assign_ops = [tf.assign(lod_grow_g_a, lod_in_grow), tf.assign(lod_grow_d_a, lod_in_grow),
                          tf.assign(lod_grow_gtmp_a, lod_in_grow), tf.assign(lod_grow_dtmp_a, lod_in_grow)]

        gan_summary = tf.summary.merge([g_summary, tf.summary.scalar('lod_in_gen', lod_g_a),tf.summary.scalar('lod_in_disc', lod_d_a),tf.summary.scalar('lod_in_grow',lod_grow_g_a)])
        #---------------------------------------------------------------

        # Morphing Vars and OPs (Reference at the bottom), only supports up to reoslution 128 
        #----------------------------------------------------------------------------------
        tvars_d = [var for var in tf.trainable_variables() if "discriminator" in var.name]
        tvars_g = [var for var in tf.trainable_variables() if "generator" in var.name]
        to_rgb_names = ["ToRGB_lod"  + num for num in ["5","4","3","2","1","0"]]
        fr_rgb_names = ["FromRGB_lod"+ num for num in ["5","4","3","2","1","0"]]
        layer_names  = ["8x8","16x16","32x32","64x64","128x128"]
        conv_names_g = ["Conv0", "Conv1"]
        conv_names_d = ["Conv0", "Conv1"]
        weight_names = ["weight", "bias"]
        
        all_rgb_vars_g = []
        all_rgb_plh_g = []
        all_rgb_asgn_ops_g = []
        all_rgb_vars_d = []
        all_rgb_plh_d = []
        all_rgb_asgn_ops_d = []
        for rgb_g, rgb_d in zip(to_rgb_names, fr_rgb_names):
            layer_asgn_ops_g = []
            layer_plh_list_g = []
            layer_var_names_g = []
            layer_asgn_ops_d = []
            layer_plh_list_d = []
            layer_var_names_d = []
            for weight in weight_names:
                var_name_g = rgb_g+"/"+weight+":0"
                var_name_d = rgb_d+"/"+weight+":0"
                pl_rgb_g = "pl_gen_"+rgb_g +"_"+ weight
                pl_rgb_d = "pl_dis_"+rgb_d +"_"+ weight
                tvar_g = [var for var in tvars_g if var_name_g in var.name][0]
                tvar_d = [var for var in tvars_d if var_name_d in var.name][0]
                pl_g   = tf.placeholder(tf.float32, name=pl_rgb_g, shape=tvar_g.shape)
                pl_d   = tf.placeholder(tf.float32, name=pl_rgb_d, shape=tvar_d.shape)
                asgn_op_g = tf.assign(tvar_g, pl_g)
                asgn_op_d = tf.assign(tvar_d, pl_d)

                layer_var_names_g.append(tvar_g)
                layer_plh_list_g.append(pl_g)
                layer_asgn_ops_g.append(asgn_op_g)
                layer_var_names_d.append(tvar_d)
                layer_plh_list_d.append(pl_d)
                layer_asgn_ops_d.append(asgn_op_d)
            
            all_rgb_vars_g.append(layer_var_names_g)
            all_rgb_plh_g.append(layer_plh_list_g)
            all_rgb_asgn_ops_g.append(layer_asgn_ops_g)
            all_rgb_vars_d.append(layer_var_names_d)
            all_rgb_plh_d.append(layer_plh_list_d)
            all_rgb_asgn_ops_d.append(layer_asgn_ops_d)

        all_var_names_g=[]
        all_plh_list_g =[]
        all_asgn_ops_g =[]
        all_var_names_d=[]
        all_plh_list_d =[]
        all_asgn_ops_d =[]

        for layer in layer_names:
            layer_asgn_ops_g = []
            layer_plh_list_g = []
            layer_var_names_g = []
            layer_asgn_ops_d = []
            layer_plh_list_d = []
            layer_var_names_d = []
            for conv_g, conv_d in zip(conv_names_g, conv_names_d):
                for weight in weight_names:
                    var_name_g = "/"+layer+"/"+conv_g+"/"+weight+":0"
                    var_name_d = "/"+layer+"/"+conv_d+"/"+weight+":0"
                    
                    pl_name_g = "pl_gen_"+layer+"_"+conv_g+"_"+weight 
                    pl_name_d = "pl_dis_"+layer+"_"+conv_d+"_"+weight

                    tvar_g = [var for var in tvars_g if var_name_g in var.name][0]
                    tvar_d = [var for var in tvars_d if var_name_d in var.name][0]

                    pl_g   = tf.placeholder(tf.float32, name=pl_name_g, shape=tvar_g.shape)
                    pl_d   = tf.placeholder(tf.float32, name=pl_name_d, shape=tvar_d.shape)
                    asgn_op_g = tf.assign(tvar_g, pl_g)
                    asgn_op_d = tf.assign(tvar_d, pl_d)
                    
                    layer_var_names_g.append(tvar_g)
                    layer_plh_list_g.append(pl_g)
                    layer_asgn_ops_g.append(asgn_op_g)
                    layer_var_names_d.append(tvar_d)
                    layer_plh_list_d.append(pl_d)
                    layer_asgn_ops_d.append(asgn_op_d)
                    
            all_var_names_g.append(layer_var_names_g)
            all_plh_list_g.append(layer_plh_list_g)
            all_asgn_ops_g.append(layer_asgn_ops_g)
            all_var_names_d.append(layer_var_names_d)
            all_plh_list_d.append(layer_plh_list_d)
            all_asgn_ops_d.append(layer_asgn_ops_d)

        # Reference:
        # 1.) RGBs:
        # w,b = all_RGB[i] = i'th layer RGBs (4x4 [0]->128x128 [5])
        # 2.) Convs:
        # Gen: w1, b1, w2, b2 = all_gen[i] = i'th layer Conv0_up and Conv1 weights and biases
        # Disc: w1, b1, w2, b2 = all_disc[i] = i'th layer Conv0, Conv1_down weights and biases
        #----------------------------------------------------------------------------------

    #----------------------------------------------------------------------------------
    #SESSION
    #----------------------------------------------------------------------------------
    with tf.Session(graph=graph) as sess:

            # Load model or initialise vars/lod 
            if load_model:
                step = restore_model(checkpoint_folder, sess)
            else:
                sess.run(tf.global_variables_initializer())
                step = 0

            # Saver, summary_op & writer, coordinator, Training Schedule
            TS = TrainingSchedule(final_resolution=input_resolution,
                    lod_training_kimg=kimg_per_phase, lod_transition_kimg=kimg_per_phase)


            lod_step_g = step
            lod_step_d = step
            lod_step_grow = step
            lod_val_g = np.floor(TS.get_lod(lod_step_g*batch_size*n_critic))
            lod_val_d = np.floor(TS.get_lod(lod_step_d*batch_size*n_critic))
            lod_val_grow = lod_val_g
            lod_new_grow = lod_val_grow
            lod_grow_bool = 0

            # Define saver, writer, etc
            saver = tf.train.Saver(max_to_keep=15)
            train_writer = tf.summary.FileWriter(checkpoint_folder, graph)
            coord = tf.train.Coordinator()

            # Ensure train init and assign current lod_vals 
            sess.run(iter_train_init_op)
            sess.run([lod_assign_ops], feed_dict={lod_in_g:lod_val_g,lod_in_d:lod_val_d})
            sess.run([lod_grow_assign_ops], feed_dict={lod_in_grow:lod_val_grow})


            # --------------------------------------------------
            # Inference (generate and save images), either final model
            # --------------------------------------------------
            if is_training==False:

                if is_testing == True:
                    all_out = []
                    sess.run(model.to_testing())
                    for i in range(inference_steps):
                        if i%50 == 0:
                            logging.info('Current inference Step: %d' % i)
                        outputs = sess.run([model.get_fake_images(g_var_scope = "generator")])
                        all_out.append(outputs)
                        if i%100 == 0 and i >0:
                            np.save(checkpoint_folder + '/inf_'+ str(i), all_out)
                            all_out = []
                    sess.run(model.to_training())
                    np.save(checkpoint_folder + '/inf_'+str(i), all_out)
                    raise ValueError('Inference done.')
            # --------------------------------------------------

            # EMA initializer to_training
            sess.run(model.to_training())

            # Train 
            while step < training_steps and not coord.should_stop():
                try:

                    dg_val,maxmin_val,minmax_val=training_step(sess,step,model,n_critic,d_train,
                        sum_freq,g_train,gan_summary,snap_freq,snap_summary,
                        dg_freq,current_to_tmp_op,iter_val_init_op,dg_steps,
                        d_train_worst,iter_test_init_op,minmax,g_train_worst,
                        maxmin,iter_train_init_op)

                    old_step = step
                    lod_step_g = step
                    lod_step_d = step

                    # Update lod
                    lod_new_g = np.floor(TS.get_lod(lod_step_g*batch_size*n_critic))
                    lod_new_d = np.floor(TS.get_lod(lod_step_d*batch_size*n_critic))

                    # Growth LOD helper to determine when to morph and to ip non-lins 
                    if step%steps_per_phase == 0 and step>0 and step%(2*steps_per_phase) != 0:
                        lod_grow_bool = 1
                    if lod_grow_bool == 1:
                        if lod_new_grow == np.floor(lod_new_grow):
                            lod_new_grow = lod_new_grow-1.0/growth_ip_steps
                        else: 
                            lod_new_grow = np.maximum(lod_new_grow-1.0/growth_ip_steps, np.floor(lod_new_grow))
                        if lod_new_grow%np.floor(lod_new_grow)<1e-3:
                            lod_new_grow = np.floor(lod_new_grow)
                            lod_grow_bool = 0
                        if lod_new_grow<0.0:
                            lod_new_grow=0.0

                    #----------------------------------------------------------------------------------
                    # MORPHISM
                    #----------------------------------------------------------------------------------
                    # Reference: 
                    # GAN Filters:   (K,K,Cin,Cout)
                    # Morph Filters: (Cout,Cin,K,K)

                    if lod_new_g != lod_val_g or lod_new_d != lod_val_d:
                        # Snaps before Morphing 
                        sum_be_morph = sess.run(snap_summary_2)
                        train_writer.add_summary(sum_be_morph, step)
                        train_writer.flush()
                        
                        # Find morph layer, and get all weight variables/references 
                        ci = int(4.0 - lod_new_g) #0,1,2,3,4
                        w_torgb_old, b_torgb_old = sess.run(all_rgb_vars_g[ci])
                        w_fromrgb_old, b_fromrgb_old = sess.run(all_rgb_vars_d[ci])
                        w_torgb_new, b_torgb_new = sess.run(all_rgb_vars_g[ci+1])
                        w_fromrgb_new, b_fromrgb_new = sess.run(all_rgb_vars_d[ci+1])
                        wtorgb_shape =  w_torgb_new.shape
                        btorgb_shape =  b_torgb_new.shape
                        wfromrgb_shape =  w_fromrgb_new.shape
                        bfromrgb_shape =  b_fromrgb_new.shape

                        w1_g_shape = all_var_names_g[ci][0].shape
                        b1_g_shape = all_var_names_g[ci][1].shape
                        w2_g_shape = all_var_names_g[ci][2].shape
                        b2_g_shape = all_var_names_g[ci][3].shape
                        w1_d_shape = all_var_names_d[ci][0].shape
                        b1_d_shape = all_var_names_d[ci][1].shape
                        w2_d_shape = all_var_names_d[ci][2].shape
                        b2_d_shape = all_var_names_d[ci][3].shape

                        # ----------------------Generator ---------------------
                        # Case 1: No halving of convolutional channels:
                        if w1_g_shape == w2_g_shape:
                            G = np.transpose(w_torgb_old, [3,2,0,1]) #(K,K,Cin,Cout)->(Cout,Cin,K,K)
                            Cout = w1_g_shape[3]
                            F1, F3 = decomp_filters_lsq_iter(G,Cout,3,1,iters=100) # morph 1 step
                            F2, F3 = decomp_filters_lsq_iter(F3,Cout,3,1,iters=100) # morph 2 step
                            
                            ## Identity Filter 
                            #F1     = np.zeros(w1_g_shape)
                            #for i in range(0,w1_g_shape[2]):
                            #    F1[:,:,i,i] = np.array([[0,0,0],[0,1.0,0],[0,0,0]])
                            
                            w1_g = np.transpose(F1,[2,3,1,0])
                            w1_g = w1_g/get_wscale(w1_g.shape) #Undo wscaling effect
                            b1_g = np.zeros(shape=b1_g_shape)
                            w2_g = np.transpose(F2, [2,3,1,0]) #(Cout,Cin,K,K)->(K,K,Cin,Cout)
                            w2_g = w2_g/get_wscale(w2_g.shape) #Undo wscaling effect
                            b2_g = np.zeros(shape=b2_g_shape)
                            wrgb_g = np.transpose(F3, [2,3,1,0]) #(Cout,Cin,K,K)->(K,K,Cin,Cout) || Here no wscale undo because same as previous toRGB
                            brgb_g = b_torgb_old

                            assert w1_g.shape == w1_g_shape and w2_g.shape == w2_g_shape
                            assert wrgb_g.shape == wtorgb_shape and brgb_g.shape == btorgb_shape

                        # Case 2: Halving of convolutional channels:
                        else:
                            G = np.transpose(w_torgb_old, [3,2,0,1]) #(K,K,Cin,Cout)->(Cout,Cin,K,K)
                            Cout = w1_g_shape[3]
                            F1, F3 = decomp_filters_lsq_iter(G,Cout,3,1,iters=100)

                            ## Identity Filter
                            #F2     = np.zeros(w2_g_shape)
                            #for i in range(0,w2_g_shape[2]):
                            #    F2[:,:,i,i] = np.array([[0,0,0],[0,1.0,0],[0,0,0]])

                            F2, F3 = decomp_filters_lsq_iter(F3,Cout,3,1,iters=100)

                            w1_g = np.transpose(F1, [2,3,1,0]) #(Cout,Cin,K,K)->(K,K,Cin,Cout)
                            w1_g = w1_g/get_wscale(w1_g.shape) #Undo wscale
                            b1_g = np.zeros(shape=b1_g_shape)
                            w2_g = np.transpose(F2, [2,3,1,0])
                            w2_g = w2_g/get_wscale(w2_g.shape) #Undo wscale
                            b2_g = np.zeros(shape=b2_g_shape)  #(Cout,Cin,K,K)->(K,K,Cin,Cout)
                            wrgb_g = np.transpose(F3, [2,3,1,0])
                            wrgb_g = wrgb_g/get_wscale(wrgb_g.shape, gain=1)*get_wscale(w_torgb_old.shape, gain=1)
                            brgb_g = b_torgb_old

                            logging.info('w2_is %s, w2_should %s', w2_g.shape, w2_g_shape)
                            logging.info('w1_is %s, w1_should %s', w1_g.shape, w1_g_shape)
                            logging.info('wrgb_is %s, wrgb_should %s', wrgb_g.shape, wtorgb_shape)
                            logging.info('brgb_is %s, brgb_should %s', brgb_g.shape, btorgb_shape)
                            assert w1_g.shape == w1_g_shape and w2_g.shape == w2_g_shape
                            assert wrgb_g.shape == wtorgb_shape and brgb_g.shape == btorgb_shape

                        # Assign new Weights:
                        sess.run([all_rgb_asgn_ops_g[ci+1][0],all_rgb_asgn_ops_g[ci+1][1]],
                            feed_dict={all_rgb_plh_g[ci+1][0]:wrgb_g,all_rgb_plh_g[ci+1][1]:brgb_g})
                        
                        sess.run([all_asgn_ops_g[ci][0],all_asgn_ops_g[ci][1],
                            all_asgn_ops_g[ci][2],all_asgn_ops_g[ci][3]], feed_dict={all_plh_list_g[ci][0]:w1_g,
                            all_plh_list_g[ci][1]:b1_g, all_plh_list_g[ci][2]:w2_g,all_plh_list_g[ci][3]:b2_g})
                        # -----------------------------------------------------
                        
                        # ----------------- Discriminator ---------------------
                        # Case 1: No Doubling of convolutional channels:
                        if w1_d_shape == w2_d_shape:
                            G = np.transpose(w_fromrgb_old, [3,2,0,1]) #(K,K,Cin,Cout)->(Cout,Cin,K,K)
                            Cout = w2_d_shape[2]
                            F1, F3 = decomp_filters_lsq_iter(G,Cout,1,3,iters=100)
                            F1, F2 = decomp_filters_lsq_iter(F1,Cout,1,3, iters=100)

                            # Identity filter (no longer) 
                            #F3     = np.zeros(w2_d_shape)
                            #for i in range(0,w2_d_shape[2]):
                            #    F3[:,:,i,i] = np.array([[0,0,0],[0,1.0,0],[0,0,0]])
                            
                            wrgb_d = np.transpose(F1, [2,3,1,0]) #(Cout,Cin,K,K)->(K,K,Cin,Cout), here no undo-wscale: Same Dim as fromRGB_old
                            brgb_d = np.zeros(bfromrgb_shape)
                            w1_d = np.transpose(F2, [2,3,1,0]) #(Cout,Cin,K,K)->(K,K,Cin,Cout)
                            w1_d = w1_d/get_wscale(w1_d.shape) #Undo wscale
                            b1_d = np.zeros(shape=b1_d_shape)
                            w2_d = np.transpose(F3,[2,3,1,0]) 
                            w2_d = w2_d/get_wscale(w2_d.shape) #Undo wscale
                            b2_d = b_fromrgb_old
                            assert w1_d.shape == w1_d_shape and w2_d.shape == w2_d_shape
                            assert wrgb_d.shape == wfromrgb_shape and brgb_d.shape == bfromrgb_shape

                        # Case 2: Doubling of convolutional channels:
                        else:
                            G = np.transpose(w_fromrgb_old, [3,2,0,1]) #(K,K,Cin,Cout)->(Cout,Cin,K,K)
                            Cout = w2_d_shape[2]
                            F1, F3 = decomp_filters_lsq_iter(G,Cout,1,3,iters=100)
                            F1, F2 = decomp_filters_lsq_iter(F1,Cout,1,3,iters=100)

                            #F2     = np.zeros(w1_d_shape)
                            #for i in range(0,w1_d_shape[2]):
                            #    F2[:,:,i,i] = np.array([[0,0,0],[0,1.0,0],[0,0,0]])
                            
                            wrgb_d = np.transpose(F1, [2,3,1,0]) #(Cout,Cin,K,K)->(K,K,Cin,Cout)
                            wrgb_d = wrgb_d/get_wscale(wrgb_d.shape,gain=1)*get_wscale(w_fromrgb_old.shape,gain=1) #Same wscale as old
                            brgb_d = np.zeros(bfromrgb_shape)
                            w1_d = np.transpose(F2, [2,3,1,0])
                            w1_d = w1_d/get_wscale(w1_d.shape)
                            b1_d = np.zeros(shape=b1_d_shape)
                            w2_d = np.transpose(F3, [2,3,1,0]) #(Cout,Cin,K,K)->(K,K,Cin,Cout)
                            w2_d = w2_d/get_wscale(w2_d.shape)
                            b2_d = b_fromrgb_old
                            logging.info('w2_is %s, w2_should %s', w2_d.shape, w2_d_shape)
                            logging.info('w1_is %s, w1_should %s', w1_d.shape, w1_d_shape)
                            logging.info('wrgb_is %s, wrgb_should %s', wrgb_d.shape, wfromrgb_shape)
                            logging.info('brgb_is %s, brgb_should %s', brgb_d.shape, bfromrgb_shape)
                            assert w1_d.shape == w1_d_shape and w2_d.shape == w2_d_shape
                            assert wrgb_d.shape == wfromrgb_shape and brgb_d.shape == bfromrgb_shape
                        # -----------------------------------------------------
                        # Assign new weights:
                        sess.run([all_rgb_asgn_ops_d[ci+1][0],all_rgb_asgn_ops_d[ci+1][1]],
                            feed_dict={all_rgb_plh_d[ci+1][0]:wrgb_d,all_rgb_plh_d[ci+1][1]:brgb_d})
                        
                        sess.run([all_asgn_ops_d[ci][0],all_asgn_ops_d[ci][1],
                            all_asgn_ops_d[ci][2],all_asgn_ops_d[ci][3]], feed_dict={all_plh_list_d[ci][0]:w1_d,
                            all_plh_list_d[ci][1]:b1_d, all_plh_list_d[ci][2]:w2_d,all_plh_list_d[ci][3]:b2_d})
                        # -----------------------------------------------------
                        # Assign new lod's 
                        sess.run([lod_assign_ops], feed_dict={lod_in_g:lod_new_g,lod_in_d:lod_new_d})
                        sess.run([g_opt_reset, d_opt_reset])

                        # Sumamry after morphing
                        sum_be_morph_3 = sess.run(snap_summary_1)
                        train_writer.add_summary(sum_be_morph_3, step)
                        train_writer.flush()

                    if lod_new_grow != lod_val_grow:
                        sess.run([lod_grow_assign_ops], feed_dict={lod_in_grow:lod_new_grow})

                    lod_val_g = lod_new_g
                    lod_val_d = lod_new_d
                    lod_val_grow = lod_new_grow

                    # Saver
                    if step > 0 and step % save_model_freq == 0:
                        save_path = saver.save(sess,
                                               checkpoint_folder + "/model.ckpt",
                                               global_step=step)
                        logging.info("Model saved in file: {}".format(save_path))
                    
                    if step == old_step:
                        step += 1
                except KeyboardInterrupt:
                    logging.info('Interrupted')
                    coord.request_stop()
                except Exception as e:
                    logging.warning(
                        "Unforeseen error at step {}. Requesting stop...".format(
                            step))
                    coord.request_stop(e)

            save_path = saver.save(sess, checkpoint_folder + "/model.ckpt",
                                   global_step=step)

            logging.info("Model saved in file: %s" % save_path)


def main(unused_argv):
    train()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()



