import logging
import codecs
from datetime import datetime

import tensorflow as tf
import numpy as np
from model import Model
from linearModel import Config
from bleu import corpus_bleu
from cam.sgnmt.decoding.beam import BeamDecoder
from cam.sgnmt.decode_utils import prepare_trg_sentences

class QModel(Model):
    """
    Implements a fully connected neural network with a single hidden layer.

    This network will predict the cumulative future rewards for each action
    (R/W) based on the inputs, which are the last decoder outputs from the
    Transformer model. (1*512 tensor for each sentence).
    """

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors

        input_holder:   Tensor of hidden states
        target_holder:  Target Q values, flatten [max_length, batch_size]
        actions_holder: Chosen actions, flatten [max_length, batch_size]
        dropout_holder: Dropout value for regularisation
        """
        self.input_holder = tf.placeholder(tf.float32, [None, self.config.d_model],
                                           name="input_holder")
        self.target_holder = tf.placeholder(tf.float32, [None])
        self.actions_holder = tf.placeholder(tf.int32, [None])
        self.dropout_holder = tf.placeholder(tf.float32, name="dropout_holder")

    def create_feed_dict(self, inputs_batch, actions=None, targets=None, dropout=1):
        """Creates the feed_dict for the linear RL agent.

        Args:
            inputs_batch:   A batch of input data.
            actions:        A batch of chosen actions [max_length*batch_size]
            targets:        Targets for qval predictions [max_length*batch_size]
            dropout:        The dropout rate.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        feed_dict = {
            self.input_holder: inputs_batch,
            self.dropout_holder: dropout
        }
        if actions is not None:
            feed_dict[self.actions_holder] = actions
            feed_dict[self.target_holder] = targets
        return feed_dict

    def add_prediction_op(self):
        """Adds the rewards prediction NN with a single hidden layer
            h = Relu(x*W + b1)
            h_drop = Dropout(h, dropout_rate)
            pred = h_drop*U + b2

        The high reward actions are chosen to expand the translation hypotheses,
        which will update the last hidden states x and hence the new predicted
        cumulative rewards.

        The predicted rewards for the chosen action at each time step are
        recorded.

        The true rewards at each time step are calculated based on the
        translation and the target translation placeholders.

        Returns:
            pred: Predicted qvals/rewards of actions
                  [batch_size*max_length, 2] --> training
                  [batch_size, 2]            --> running
        """

        with tf.variable_scope("linear"):
            W = tf.get_variable('W', [self.config.d_model, self.config.hidden_size],
                            tf.float32, tf.contrib.layers.xavier_initializer())
            b1 = tf.get_variable('b1', [self.config.hidden_size],
                            tf.float32, tf.constant_initializer(0))
            U = tf.get_variable('U', [self.config.hidden_size, 2],
                            tf.float32, tf.contrib.layers.xavier_initializer())
            b2 = tf.get_variable('b2', [2],
                            tf.float32, tf.constant_initializer(0))

            h = tf.nn.relu(tf.matmul(self.input_holder, W) + b1)
            h_drop = tf.nn.dropout(h, self.dropout_holder)
            self.preds = tf.add(tf.matmul(h_drop, U), b2, name="predictions")

        return self.preds

    def add_loss_op(self, probs_history):
        """Adds Ops for the loss function to the computational graph.
        Use the mask to mask out nan produced when evaluating log(0)*0, since
        the last few elements of probs_holder will be 0's for each sentence.

        Args:
            probs_history: Predicted probabilities of actions in a batch at all
                           time steps, shape [batch_size*max_length, 2]
        Returns:
            loss: A 0-d tensor (scalar)
        """
        zero = tf.constant(0, dtype=tf.float32)
        mask = tf.not_equal(self.target_holder, zero)
        indices = tf.stack(
            [tf.range(tf.shape(probs_history)[0]), self.actions_holder],
            axis=1 ) # which element to pick in 2D array
        chosen_actions = tf.gather_nd(probs_history, indices)
        raw_loss = tf.square(self.target_holder - chosen_actions)
        loss = tf.reduce_sum(tf.boolean_mask(raw_loss, mask))
        return loss


    def populate_train_dict(self, sess, targets_batch):
        """Prepares the feed dictionary for training.
        Args:
            targets_batch:  Target sentences in ids [batch_size, max_length]
        Returns:
            train_dict:     Feed dictionary for training
        """
        hidden_states = np.zeros(
            (self.config.max_length, len(self.cur_hypos), self.config.d_model),
            dtype=np.float32 )
        actions = np.zeros((self.config.max_length, len(self.cur_hypos)),
                           dtype=np.int32)
        targets = np.zeros((self.config.max_length, len(self.cur_hypos)),
                           dtype=np.int32)
        for step in range(self.config.max_length-1):# -1 since already have a READ
            #prev_lengths = [len(x[0].actions) for x in self.cur_hypos]
            hidden_states[step,:,:] = self._get_hidden_states()
            # current best qval is the target for the previous step
            actions[step,:], qvals, optimal_actions = self.predict_one_step(sess,
                                                       hidden_states[step,:,:])
            
            # If the target network exists, decouple the action selection from
            # target values prediction
            if hasattr(self, 'target'):
                targets[step-1,:] = self.target._get_targets(sess,
                                    hidden_states[step,:,:], optimal_actions)
            else:
                targets[step-1,:] = np.choose(optimal_actions, qvals.T)
            self._update_hidden_states(actions[step,:])

        BLEU_hypos = []
        BLEU_refs = []
        wue_BLEU_hypos = []  # the hypo holder for WUE decoding
        # Generate full hypotheses from partial hypotheses
        for idx, hypo in enumerate(self.cur_hypos):
            self.cur_hypos[idx] = hypo[0].generate_full_hypothesis()
            action_length = len(self.cur_hypos[idx].actions)
            # get hypothesis and reference for BLEU evaluation
            BLEU_hypos.append([str(x) for x in self.cur_hypos[idx].trgt_sentence])
            BLEU_refs.append([[str(x) for x in targets_batch[idx]]])
            if hasattr(self, 'target') and self.config.useBLEUDrop:
                wue_BLEU_hypos.append([str(x) for x in 
                                self.all_wue_trans[self.cur_hypos[idx].lst_id]])
            # targets are just as long as the action sequence, pad the remaining
            # targets with 0
            targets[action_length-2:,idx] = 0

        # give the quality rewards (BLEU) at the end
        _, quality = corpus_bleu(BLEU_refs, BLEU_hypos) # BLEU score for the batch
        
        if self.config.useBLEUDrop:
            _, wue_BLEU = corpus_bleu(BLEU_refs, wue_BLEU_hypos)
            quality = quality - wue_BLEU
        batch_average_delay = 0.0
        for idx in range(len(self.cur_hypos)):
            batch_average_delay += self.cur_hypos[idx].get_average_delay()
            targets[len(self.cur_hypos[idx].actions)-2,idx] = \
                quality + self.cur_hypos[idx].get_last_delay_reward(self.config)

        batch_average_delay /= len(self.cur_hypos)
        logging.info("\n     batch average delay: %f\n" % batch_average_delay)
        logging.info("\n     batch (delta) BLEU:  %f\n" % quality)
        if self.config.useBLEUDrop:
            logging.info("\n     batch WUE BLEU:      %f\n" % wue_BLEU)
        train_dict = self.create_feed_dict(
            np.reshape(hidden_states, (-1, self.config.d_model)),
            actions.flatten(),
            targets.flatten(),
            self.config.dropout )

        return train_dict


    def _get_targets(self, sess, inputs_batch, actions, dropout=None):
        """Make one step estimation of the batch target given the actions,
        using the target network in DQN.

        Returns:
            targets:   probs/rewards of chosen actions, [1, batch_size]
        """
        if dropout is None:
            dropout = self.config.dropout
        feed = self.create_feed_dict(inputs_batch, dropout=dropout)
        predictions = sess.run(self.pred, feed_dict=feed)
        targets = np.choose(actions, predictions.T)
        
        return targets


    def __init__(self, config):
        super(QModel, self).__init__(config.args, config.isTargetNet)
        self.config = config
        self.build()
        if config.useBLEUDrop and not config.isTargetNet:
            with codecs.open(config.args.trg_trans, encoding='utf-8') as f:
                self.all_wue_trans = prepare_trg_sentences(
                                         [line.strip().split() for line in f])

