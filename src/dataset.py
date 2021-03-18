import tensorflow as tf 
import os
import multiprocessing

class Dataset:
    """
    Base Dataset.

    Call get_dataset() to get this dataset.
    """

    def __init__(
            self, 
            path_input,
            new_size = 128, 
            batch_size = 64,
            shuffle=True,
            is_prefetch = True, 
            buffer_size=10, 
            testing=False,
            repeat=True, 
            num_parallel_calls=None, 
            drop_remainder=True,
            what_dataset = "celeba"): #"celeba" or "lsun" 
    
            """
        Args:
            path_input: path to input txt file
            crop: Aspect face/background
            orig_size: Shape of images in dataset
            size: New Shape of images [size,size]
            batch_size: int
            shuffle: whether to shuffle input samples
            buffer_size: size of buffer (in batches)
            testing: whether we are testing or not
            repeat: whether to iterate through the dataset several times
            num_parallel_calls: number of threads to use for preprocessing
            drop_remainder: if True we never yield incomplete batches
            sentence_length: length of sentences
            vocabulary_size: number of words in vocabulary
        """
            #orig_size = [218,178]
            #self.orig_size = orig_size
            #crop = 150

            self.path_input = path_input

            if what_dataset == "celeba":
                self.crop = 150
                self.orig_shape = [218,178]
            elif what_dataset == "lsun":
                self.crop = 200
                self.orig_shape = [256,360]
                
            self.new_size = new_size 
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.is_prefetch = is_prefetch
            self.buffer_size = buffer_size
            self.testing = testing
            self.repeat = repeat
            self.drop_remainder = drop_remainder
            self.num_parallel_calls = num_parallel_calls \
                if num_parallel_calls \
                   is not None else multiprocessing.cpu_count() - 1
            self.drop_remainder = drop_remainder
            self.what_dataset = what_dataset
            self.all_paths = [os.path.join(self.path_input,path) for path in os.listdir(self.path_input)]

    def _prepare_iterator(self, dataset):
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=
                                      self.buffer_size * self.batch_size)

        if self.is_prefetch:
            dataset = dataset.prefetch(self.buffer_size)

        if self.repeat:
            dataset = dataset.repeat()

        if self.drop_remainder:
            dataset = dataset.batch(self.batch_size,
                                drop_remainder=True)

        #dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size))
        #dataset = dataset.prefetch(self.buffer_size)

        return dataset
        
    def load_and_preprocess_image(self,path):
        if self.what_dataset == "celeba":
            image = tf.image.decode_jpeg(tf.read_file(path), channels=3)
        elif self.what_dataset == "lsun":
            image = tf.image.decode_image(tf.read_file(path), channels=3)

        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        #image = tf.image.resize_images(image, self.orig_shape)
        image = tf.image.resize_image_with_crop_or_pad(image, self.orig_shape[0], self.orig_shape[1])
        j = (int(image.shape[0]) - self.crop) // 2
        i = (int(image.shape[1]) - self.crop) // 2
        image = tf.image.crop_to_bounding_box(image, j, i, self.crop, self.crop)
        image = tf.image.resize_images(image, [self.new_size,self.new_size], method=tf.image.ResizeMethod.BILINEAR)
        return tf.transpose(image, [2,0,1])

    def get_dataset(self):
        """
        Creates an iterator for dataset specified by self.path_input
        Returns: iterator returns NCHW format [batch_size,3,new_size,new_size]

        """
        path_ds = tf.data.Dataset.from_tensor_slices(tf.constant(self.all_paths))
        image_ds = path_ds.map(self.load_and_preprocess_image, 
            num_parallel_calls=self.num_parallel_calls)

        dataset = self._prepare_iterator(image_ds)
        return dataset
