import os
import sys
import time
import logging
from datetime import datetime

import tensorflow as tf
import numpy as np
from tensorflow.python.training import training

from linearModel import Config
from linearModel import linearModel
from QModel import QModel
from cam.sgnmt import utils
from cam.sgnmt.ui import get_args, validate_args

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s')
logging.getLogger().setLevel(logging.DEBUG)

def do_train(args, shuffle=True):
    """Main training function"""
    with tf.Graph().as_default():
        logging.info("Building model...",)
        start = time.time()
        linModel = linearModel(config)
        logging.info("took %.2f seconds", time.time() - start)

        train_size = len(linModel.all_hypos)
        indices = np.arange(train_size)
        assert len(linModel.all_hypos) == len(linModel.all_trg)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            for i in range(config.n_epochs):
                if shuffle:
                    np.random.shuffle(indices)
                if i > 0:
                    linModel.prepareSGNMT(args) # reset hypos

                for batch_start in np.arange(0, train_size, config.batch_size):
                    batch_indices = indices[batch_start: batch_start+config.batch_size]
                    linModel.cur_hypos = [linModel.all_hypos[i] for i in batch_indices]
                    targets_batch = [linModel.all_trg[i] for i in batch_indices]
                    loss = linModel.train_on_batch(session, targets_batch)
                    logging.info("progress: %d/%d" % (batch_start, train_size))
                    logging.info("loss: %d\n" % loss)

                linModel.config.eps = 0.5*linModel.config.eps # annealing
                # save the model every epoch
                saver.save(session, config.output_path + "RLmodel", global_step=i)

def do_single_epoch_train(args, shuffle=True):
    """Main training function"""
    with tf.Graph().as_default():
        logging.info("Building model...",)
        start = time.time()
        linModel = QModel(config)
        logging.info("took %.2f seconds", time.time() - start)

        train_size = len(linModel.all_hypos)
        indices = np.arange(train_size)
        assert len(linModel.all_hypos) == len(linModel.all_trg)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        #with linModel.create_session() as session:
        with create_training_session(config) as session:
            #session.run(init)
            if shuffle:
                np.random.shuffle(indices)

            for nb in range(config.n_batches):
                batch_start = nb*config.batch_size
                batch_indices = indices[batch_start: batch_start+config.batch_size]
                linModel.cur_hypos = [linModel.all_hypos[i] for i in batch_indices]
                targets_batch = [linModel.all_trg[i] for i in batch_indices]
                loss = linModel.train_on_batch(session, targets_batch)
                logging.info("progress: %d/%d" % (batch_start, train_size))
                logging.info("eps: %f" % linModel.config.eps)
                logging.info("loss: %f\n" % loss)
                # monitored training session handles the checkpoint saving
                if nb % 6 == 0 and nb != 0:
                    linModel.config.eps = max(config.min_eps, 0.95*linModel.config.eps)
                    #saver.save(session, config.output_path + "RLmodel", global_step=i)

def do_multi_epoch_train(args, shuffle=True):
    """Main training function"""
    with tf.Graph().as_default():
        logging.info("Building model...",)
        start = time.time()
        linModel = linearModel(config)
        logging.info("took %.2f seconds", time.time() - start)

        train_size = len(linModel.all_hypos)
        indices = np.arange(train_size)
        assert len(linModel.all_hypos) == len(linModel.all_trg)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        if shuffle:
            np.random.shuffle(indices)
        # select the training subset from the file
        indices = indices[:config.batch_size*config.n_batches]
        batch_count = 0
        #with linModel.create_session() as session:
        with create_training_session(config) as session:
            for current_epoch in range(config.n_epochs):
                if shuffle:
                    np.random.shuffle(indices)

                for nb in range(config.n_batches):
                    batch_count += 1
                    batch_start = nb*config.batch_size
                    batch_indices = indices[batch_start: batch_start+config.batch_size]
                    linModel.cur_hypos = [linModel.all_hypos[i] for i in batch_indices]
                    targets_batch = [linModel.all_trg[i] for i in batch_indices]
                    loss = linModel.train_on_batch(session, targets_batch)
                    logging.info("epoch: %d" % (current_epoch+1))
                    logging.info("progress: %d/%d" % (batch_start, train_size))
                    logging.info("eps: %f" % linModel.config.eps)
                    logging.info("loss: %f\n" % loss)
                    # monitored training session handles the checkpoint saving
                    if batch_count % 6 == 0:
                        # anneal according to global batch count
                        linModel.config.eps = max(config.min_eps, 0.95*linModel.config.eps)
                linModel._load_all_initial_hypos(args) # reset all hypotheses

def do_DQN_train(args, shuffle=True):
    """Main training function"""

    def updateTargetGraph(ops_holder,sess):
        for op in ops_holder:
            sess.run(op)

    with tf.Graph().as_default():
        start = time.time()
        primary = QModel(config)
        with tf.variable_scope("target"):
            # put the target network in a different scope
            config.isTargetNet = True
            target = QModel(config)
        primary.target = target
        train_size = len(primary.all_hypos)
        indices = np.arange(train_size)
        assert len(primary.all_hypos) == len(primary.all_trg)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        # define procedure to update the target graph
        trainables = tf.trainable_variables()
        num_var = len(trainables)
        op_holder = []
        for idx,var in enumerate(trainables[0:num_var//2]):
            op_holder.append(trainables[idx+num_var//2].assign(
                (var.value()*config.tau) + \
                ((1-config.tau)*trainables[idx+num_var//2].value())))

        if shuffle:
            np.random.shuffle(indices)
        # select the training subset from the file
        indices = indices[:config.batch_size*config.n_batches]
        batch_count = 0

        with create_training_session(config) as session:
            for current_epoch in range(config.n_epochs):
                if shuffle:
                    np.random.shuffle(indices)

                for nb in range(config.n_batches):
                    batch_count += 1
                    batch_start = nb*config.batch_size
                    batch_indices = indices[batch_start: batch_start+config.batch_size]
                    primary.cur_hypos = [primary.all_hypos[i] for i in batch_indices]
                    targets_batch = [primary.all_trg[i] for i in batch_indices]

                    loss = primary.train_on_batch(session, targets_batch)
                    updateTargetGraph(op_holder, session)

                    logging.info("epoch: %d" % (current_epoch+1))
                    logging.info("progress: %d/%d" % (batch_start, train_size))
                    logging.info("eps: %f" % primary.config.eps)
                    logging.info("loss: %f\n" % loss)
                    # monitored training session handles the checkpoint saving
                    if batch_count % 6 == 0:
                        # anneal according to global batch count
                        primary.config.eps = max(config.min_eps, 0.95*primary.config.eps)
                primary._load_all_initial_hypos(args) # reset all hypotheses


def debug_train(args):
    """Training script for debug only"""
    with tf.Graph().as_default():
        logging.info("Building model...",)
        start = time.time()
        linModel = linearModel(config)
        logging.info("took %.2f seconds", time.time() - start)

        linModel.cur_hypos = linModel.all_hypos[5:5+linModel.config.batch_size]
        targets_batch = linModel.all_trg[5:5+linModel.config.batch_size]
        init = tf.global_variables_initializer()
        logging.info("Source 6 length %d" % len(linModel.all_src[5]))
        logging.info("Source 7 length %d" % len(linModel.all_src[6]))

        with tf.Session() as session:
            session.run(init)
            for i in range(config.n_epochs):
                loss = linModel.train_on_batch(session, targets_batch)
                logging.info("loss: %d" % loss)
                linModel.config.eps = 0.5*linModel.config.eps
                linModel.prepareSGNMT(args) # reset hypos
                linModel.cur_hypos = linModel.all_hypos[5:5+linModel.config.batch_size]
                targets_batch = linModel.all_trg[5:5+linModel.config.batch_size]


def create_training_session(config):
    """Creates a MonitoredTrainingSession for training"""
    return training.MonitoredTrainingSession(
            checkpoint_dir=config.output_path,
            save_checkpoint_secs=1200)

if __name__ == "__main__":
    # MAIN CODE STARTS HERE
    # Load configuration from command line arguments or configuration file
    args = get_args()
    validate_args(args)
    utils.switch_to_t2t_indexing()
    config = Config(args)

    do_multi_epoch_train(args, False)
