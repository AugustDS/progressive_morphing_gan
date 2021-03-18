import logging
import re

import tensorflow as tf
from datetime import datetime


def restore_model(path, sess, variables_scope=None, is_model_path=False):
    """
    Restores a model from path
    Args:
        path: where the model is stored.
            Should be a folder or a file like ...model-76739
        sess: tensorflow session
        variables_scope: we only restore variables in this scope
        is_model_path: Set to True if path is a folder

    Returns:

    """
    variables_can_be_restored = set(
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                          scope=variables_scope))
    logging.info("Loading model from '{}'".format(path))
    if is_model_path:
        # Path is something like ...model-76739
        restore_path = path
    else:
        # Path is checkpoint folder, e.g. 20190107-1032_gazenet_u_augmented_bw/
        checkpoint = tf.train.get_checkpoint_state(path)
        restore_path = checkpoint.model_checkpoint_path
    restore = tf.train.Saver(variables_can_be_restored)
    restore.restore(sess, restore_path)
    step = int(re.sub(r'[^\d]', '', restore_path.split('-')[-1]))
    return step


def construct_model_name(basename, version):
    current_time = datetime.now().strftime("%Y%m%d-%H%M")
    return "{}_{}_{}".format(basename, version, current_time)
