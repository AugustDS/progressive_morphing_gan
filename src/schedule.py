import numpy as np

class TrainingSchedule:
    def __init__(
        self,
        lod_initial_resolution  = 4,        # Image resolution used at the beginning.
        final_resolution        = 128,      # Final resolution used. 
        lod_training_kimg       = 150,      # Thousands of real images to show before doubling the resolution.
        lod_transition_kimg     = 150,      # Thousands of real images to show when fading in new layers.
        minibatch_base          = 16,       # Maximum minibatch size, divided evenly among GPUs.
        minibatch_dict          = {},       # Resolution-specific overrides.
        max_minibatch_per_gpu   = {},       # Resolution-specific maximum minibatch size per GPU.
        G_lrate_base            = 0.001,    # Learning rate for the generator.
        G_lrate_dict            = {},       # Resolution-specific overrides.
        D_lrate_base            = 0.001,    # Learning rate for the discriminator.
        D_lrate_dict            = {},       # Resolution-specific overrides.
        tick_kimg_base          = 160,      # Default interval of progress snapshots.
        step_to_increase        = 0,        # For dynamic step change 
        tick_kimg_dict          = {4: 160, 8:140, 16:120, 32:100, 64:80, 128:60, 256:40, 512:20, 1024:10}): # Resolution-specific overrides.
        
        self.lod_initial_resolution = lod_initial_resolution
        self.lod_training_kimg = lod_training_kimg
        self.lod_transition_kimg = lod_transition_kimg
        self.final_resolution = final_resolution
        self.lod_training_kimg = lod_training_kimg
        self.step_to_increase = step_to_increase

    def get_lod(self, cur_nimg):
        # Training phase.
        lod_final_resolution   = np.floor(np.log2(self.final_resolution))
        self.kimg = (cur_nimg+self.step_to_increase) / 1000.0
        phase_dur = self.lod_training_kimg + self.lod_transition_kimg
        phase_idx = int(np.floor(self.kimg / phase_dur)) if phase_dur > 0 else 0
        phase_kimg = self.kimg - phase_idx * phase_dur

        # Level-of-detail and resolution.
        self.lod =  lod_final_resolution
        self.lod -= np.floor(np.log2(self.lod_initial_resolution))
        self.lod -= phase_idx
        if self.lod_transition_kimg > 0:
            self.lod -= max(phase_kimg - self.lod_training_kimg, 0.0) / self.lod_transition_kimg
        self.lod = max(self.lod, 0.0)
        self.resolution = 2 ** (lod_final_resolution - int(np.floor(self.lod)))

        return self.lod
    
        # Minibatch size.
        '''
        self.minibatch = minibatch_dict.get(self.resolution, minibatch_base)
        self.minibatch -= self.minibatch % config.num_gpus
        if self.resolution in max_minibatch_per_gpu:
            self.minibatch = min(self.minibatch, max_minibatch_per_gpu[self.resolution] * config.num_gpus)

        # Other parameters.
        self.G_lrate = G_lrate_dict.get(self.resolution, G_lrate_base)
        self.D_lrate = D_lrate_dict.get(self.resolution, D_lrate_base)
        self.tick_kimg = tick_kimg_dict.get(self.resolution, tick_kimg_base)
        
TS = TrainingSchedule()
print(TS.get_lod(189400))
print(TS.get_lod(400535))

i = [150.0,160.0,170.0,180.0,190.0,200.0,210.0,220.0]
kimg_per_phase = 37500*16*1/1000.0
TS = TrainingSchedule(final_resolution=128,
                        lod_training_kimg=kimg_per_phase, 
                        lod_transition_kimg=kimg_per_phase)
TS_2 = TrainingSchedule(final_resolution=128,
                        lod_training_kimg=kimg_per_phase, 
                        lod_transition_kimg=kimg_per_phase,
                        step_to_increase = 0)

TS_2.step_to_increase = 180000*16-150000*16
TS_2.lod_training_kimg = 45000*16*1/1000.0
TS_2.lod_transition_kimg = 45000*16*1/1000.0
for j in i:
    print(j)
    l1 = TS.get_lod(j*1000*16*1)
    print(l1)
    l2 = TS_2.get_lod(j*16*1*1000)
    print(l2)
    print("------")
'''
