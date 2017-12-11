import codecs
import logging
import os
import sys
import copy

import numpy as np
import tensorflow as tf

from bleu import get_incremental_BLEU
from cam.sgnmt import utils
from cam.sgnmt import decode_utils

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
        """Sets up the traadd_loss_opining Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train.

        Args:
            loss: Loss tensor.
        Returns:
            train_op: The Op for training.
        """
        train_op = tf.train.AdamOptimizer(self.config.lr).minimize(loss)
        return train_op

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

    def predict_one_step(self, sess, inputs_batch, dropout=None):
        """Make one step prediction on the batch given the hidden states.

        Returns:
            probs:   Probabilities of chosen actions, [batch_size, 1]
            actions: Chosen actions, [batch_size, 1]
                     0 -> READ
                     1 -> WRITE
        """
        if dropout is None:
            dropout = self.config.dropout
        feed = self.create_feed_dict(inputs_batch, dropout=dropout)
        predictions = sess.run(self.pred, feed_dict=feed)
        probs = sess.run(tf.reduce_max(predictions, axis=1))
        actions = sess.run(tf.argmax(predictions, axis=1))

        return self._check_actions(actions), probs

    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)

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
        with codecs.open(args.src_train, encoding='utf-8') as f:
            self.all_hypos, self.all_src = decode_utils.prepare_sim_decode(
                    self.decoder, outputs, [line.strip().split() for line in f])

        with codecs.open(args.trg_train, encoding='utf-8') as f:
            self.all_trg = decode_utils.prepare_trg_sentences(
                                        [line.strip().split() for line in f])

    def _check_actions(self, actions):
        """Check if the actions are valid, if not, change the action."""

        for sentence in range(actions.shape[0]):
            if actions[sentence] > 0.5 and self.cur_hypos[sentence][0].netRead < 1:
                actions[sentence] = 0 # able to WRITE? (ie written < read)

            if actions[sentence] < 0.5 and self.cur_hypos[sentence][0].progress \
                        >= len(self.all_src[self.cur_hypos[sentence][0].lst_id]):
                actions[sentence] = 1 # able to READ? (ie progress < length)
        return actions

    def _update_hidden_states(self, actions):
        """Update the current hypotheses in the current batch ``self.cur_hypos''
        using actions, which will update the hiddens states.

        Args:
            actions:  np array of actions that defines how to expand each hypo,
                      update hidden states through R/W

        """
        for sentence in range(len(self.cur_hypos)):
            hypo = self.cur_hypos[sentence][0]
            if not self.decoder.stop_criterion([hypo]):
                continue # decoding finished
            self.decoder.set_predictor_states(
                                        copy.deepcopy(hypo.predictor_states) )

            if actions[sentence] < 0.5: # READ
                #logging.info("List range: %d/%d" % (hypo.lst_id, len(self.all_src)))
                #logging.info("Sentence length: %d/%d \n" %
                #                (hypo.progress, len(self.all_src[hypo.lst_id])))
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

    def _get_bacth_cumulative_rewards(self, targets_batch):
        """Calculate the Cumulative rewards for the hypotheses stored in
        ``self.cur_hypos'', where the rewards encode the delay and
        the quality.

        Args:
            targets_batch:  A batch of target data (correct translation).
        Returns:
            cum_rewards:    Cumulative rewards for all current hypotheses,
                            numpy array of shape [max_length, batch_size]
        """
        cum_rewards = np.zeros((self.config.max_length, targets_batch.shape[0]),
                               dtype=np.float32)
        for i in range(targets_batch.shape[0]):
            cum_rewards[:,i] = self.cur_hypos[i].get_delay_rewards(self.config)
            # find indices of writes in the action list
            indices = [k for k, x in enumerate(self.cur_hypos[i].actions) if x == "w"]
            # check the length of tranlstion hypothesis is the same as number of "w"
            assert len(indices) == len(self.cur_hypos[i].trgt_sentence)

            cum_reward[indices], _ += get_incremental_BLEU(
                            self.cur_hypos[i].trgt_sentence, targets_batch[i])

        return cum_rewards

    def _get_cumulative_rewards(self, index, target):
        """Calculate the Cumulative rewards for the hypothesis stored at
        ``self.cur_hypos[index]'', where the rewards encode the delay and
        the quality.

        Args:
            index:      Index of the hypothesis.
            target:     A tensor of target data (correct translation).
        Returns:
            cum_reward:     Cumulative rewards for the current hypothesis,
                            numpy array of shape [1, max_length]
        """
        cum_reward = self.cur_hypos[index].get_delay_rewards(self.config)
        # find indices of writes in the action list
        indices = [i for i, x in enumerate(self.cur_hypos[index].actions) if x == "w"]
        # check the length of tranlstion hypothesis is the same as number of "w"
        assert len(indices) == len(self.cur_hypos[index].trgt_sentence)

        cum_reward[indices], _ += get_incremental_BLEU(
                                self.cur_hypos[index].trgt_sentence, target)

        return cum_reward
