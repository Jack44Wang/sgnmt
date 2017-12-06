import codecs
import logging
import os
import sys
import copy

import numpy as np

from cam.sgnmt import utils
from cam.sgnmt import decode_utils
from cam.sgnmt.ui import get_args, validate_args

try:
    # Requires tensor2tensor
    from tensor2tensor.data_generators import text_encoder
except ImportError:
    pass 


class Model(object):
    """Abstracts a Tensorflow graph for a learning task. """

    def __init__(self, args):
        self.prepareSGNMT(args) # Initialise the SGNMT for hidden states extraction

    def add_placeholders(self):
        """Adds placeholder variables to tensorflow computational graph. """
        raise NotImplementedError("Each Model must re-implement this method.")

    def create_feed_dict(self, inputs_batch, targets_batch=None):
        """Creates the feed_dict for one step of training.

        A feed_dict takes the form of:
        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        If targets_batch is None, then no labels are added to feed_dict.
        The keys for the feed_dict should be a subset of the placeholder
        tensors created in add_placeholders.

        Args:
            inputs_batch:  A batch of input data.
            targets_batch: A batch of label data.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_prediction_op(self):
        """Implements the core of the model that transforms a batch of input
        data into predictions.

        Returns:
            pred: A tensor of shape (batch_size, n_classes)
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_loss_op(self, pred):
        """Adds Ops for the loss function to the computational graph.

        Args:
            pred: A tensor of shape (batch_size, n_classes)
        Returns:
            loss: A 0-d tensor (scalar) output
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        sess.run() to train the model.

        Args:
            loss: Loss tensor (a scalar).
        Returns:
            train_op: The Op for training.
        """

        raise NotImplementedError("Each Model must re-implement this method.")

    def train_on_batch(self, sess, inputs_batch, targets_batch):
        """Perform one step of gradient descent on the provided batch of data.

        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, d_model)
            targets_batch: np.ndarray of shape (n_samples, d_model)
        Returns:
            loss: loss over the batch (a scalar)
        """
        feed = self.create_feed_dict(inputs_batch, targets_batch=targets_batch)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def predict_on_batch(self, sess, inputs_batch):
        """Make predictions (cumulative future rewards for R/W) for the provided
        batch of data

        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, d_model)
        Returns:
            predictions: np.ndarray of shape (n_samples, 2)
        """
        feed = self.create_feed_dict(inputs_batch)
        predictions = sess.run(self.pred, feed_dict=feed)
        return predictions

    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        #self.loss = self.add_loss_op()
        #self.train_op = self.add_training_op(self.loss)

    def prepareSGNMT(self, args):
        """Initialise the SGNMT for agent training.
        The SGNMT returns hiddens states only when the function
        ``_get_hidden_states()'' is being called.
        It should stop after reading the first words, and returns the predictor
        """
        # UTF-8 support
        if sys.version_info < (3, 0):
            sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)
            sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
            sys.stdin = codecs.getreader('UTF-8')(sys.stdin)

        utils.load_src_wmap(args.src_wmap)
        utils.load_trg_wmap(args.trg_wmap)
        utils.load_trg_cmap(args.trg_cmap)
        self.decoder = decode_utils.create_decoder(args)
        self.predictor = self.decoder.predictors[0][0]# only sim_t2t predictor
        outputs = decode_utils.create_output_handlers()

        # set up SGNMT
        with codecs.open(args.src_test, encoding='utf-8') as f:
            self.all_hypos, self.all_src = decode_utils.prepare_sim_decode(
                    self.decoder, outputs, [line.strip().split() for line in f])

    def _update_hidden_states(self, actions):
        """Update the current hypotheses in the current batch ``self.cur_hypos''
        using actions, which will update the hiddens states.

        Args:
            actions:  np array of actions that defines how to expand each hypo,
                      update hidden states through R/W

        """
        for sentence in range(len(self.cur_hypos)):
            hypo = self.cur_hypos[sentence][0]
            self.decoder.set_predictor_states(
                                        copy.deepcopy(hypo.predictor_states) )
            if actions[sentence] < 0.5: # READ
                self.decoder._reveal_source_word(
                            self.all_src[hypo.lst_id][hypo.progress], [hypo] )
                hypo.progress += 1
                hypo.netRead += 1
                hypo.append_action()
                if hypo.progress == len(self.all_src[hypo.lst_id]): # reach EOS
                    self.decoder._reveal_source_word(text_encoder.EOS_ID,
                                                     [hypo] )
            else: # WRITE
                self.decoder._write_step([hypo])


    def _get_hidden_states(self):
        """Get the hidden states in the current batch.

        Returns:
            h_states: np array of shape [batch_size, d_model]
                      first dim might be small than batch_size, depends on the
                      size of ``self.cur_hypos''
        """
        h_states = np.zeros((len(self.cur_hypos), self.config.d_model), 
                            dtype=np.float32)
        for sentence in range(h_states.shape[0]):
            hypo = self.cur_hypos[sentence][0]
            self.decoder.set_predictor_states(
                                        copy.deepcopy(hypo.predictor_states) )
            h_states[sentence,:] = self.predictor.get_last_decoder_state()
        return h_states

    def _get_bacth_cumulative_rewards(self, targets_batch, mask_batch):
        """ See ``_get_cumulative_rewards()'' """
        cum_rewards = np.zeros((targets_batch.shape[0], self.config.max_length),
                               dtype=np.float32)
        for i in range(targets_batch.shape[0]):
            cum_rewards[i,:] = self._get_cumulative_rewards(i,
                                                targets_batch[i], mask_batch[i])

        return cum_rewards

    def _get_cumulative_rewards(self, index, target, mask):
        """Calculate the Cumulative rewards for the hypothesis stored at
        ``self.cur_hypos[index]'', where the rewards encode the delay and
        the quality.

        Args:
            index:      Index of the hypothesis.
            target:     A tensor of target data (correct translation).
            mask:       A tensor of mask data to the target (encodes length).

        Returns:
            cum_reward:     Cumulative rewards for the current hypothesis,
                            numpy array of shape [1, max_length]
        """
        cum_reward = np.ones((1, self.config.max_length), dtype=np.float32)
        # TODO calculate the cum_reward based on delay and quality

        return cum_reward
