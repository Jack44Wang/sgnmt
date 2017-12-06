import os
import sys
import time
import logging
from datetime import datetime

import tensorflow as tf
import numpy as np

from linearModel import Config
from linearModel import linearModel
from cam.sgnmt import utils
from cam.sgnmt.ui import get_args, validate_args

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s')
logging.getLogger().setLevel(logging.DEBUG)

def do_train(args):
    config = Config(args)

    with tf.Graph().as_default():
        logging.info("Building model...",)
        start = time.time()
        linModel = linearModel(config)
        logging.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)


def do_evaluate(args):
    config = Config(args)
    with tf.Graph().as_default():
        logging.info("Building model...",)
        start = time.time()
        linModel = linearModel(config)
        logging.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()

        with tf.Session() as session:
            session.run(init)

if __name__ == "__main__":
    # MAIN CODE STARTS HERE
    # Load configuration from command line arguments or configuration file
    args = get_args()
    validate_args(args)
    utils.switch_to_t2t_indexing()
    config = Config(args)
    with tf.Graph().as_default():
        logging.info("Building model...",)
        start = time.time()
        linModel = linearModel(config)
        logging.info("took %.2f seconds", time.time() - start)

        linModel.cur_hypos = linModel.all_hypos[:linModel.config.batch_size]
        init = tf.global_variables_initializer()

        targets_batch = np.zeros((config.batch_size, config.max_length),
                                 dtype=np.float32)
        _mask_batch = np.ones((config.batch_size, config.max_length))
        mask_batch = np.ma.make_mask(_mask_batch, shrink=False)
        with tf.Session() as session:
            session.run(init)
            linModel.train_on_batch(session, linModel._get_hidden_states(),
                                    targets_batch, mask_batch )
