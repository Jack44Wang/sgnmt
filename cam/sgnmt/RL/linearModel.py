import logging
from datetime import datetime

import tensorflow as tf
import numpy as np
from model import Model
from bleu import corpus_bleu


class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    d_model = 512
    max_length = 100 # longest sequence of actions (R/W)
    dropout = 0.5
    hidden_size = 64
    batch_size = 256  #32/128
    n_epochs = 3
    n_batches = 256 #1024/256
    lr = 0.001
    eps = 1.0       # initial probability of choosing random action
    min_eps = 0.05 # minimum probability of choosing random action
    tau = 0.01    # how much is the target graph updated to the main graph
    isTargetNet = False
    useBLEUDrop = True  # use the reduction in BLEU as quality measure

    c_trg = 18       # target consecutive delay
    d_trg = 0.8     # target average proportion
    alpha = 0.0   # for consecutive delay
    beta = -0.2     # for average proportion -0.4 for Q

    def __init__(self, args):
        self.args = args

        if args is not None and "model_path" in args:
            # Where to save things.
            self.output_path = args.model_path
        else:
            self.output_path = "/data/mifs_scratch/zw296/exp/t2t/jaen-wat/RL_train_17/"
        self.model_output = self.output_path + "model.weights"
        self.log_output = self.output_path + "log"


class linearModel(Model):
    """
    Implements a fully connected neural network with a single hidden layer.

    This network will predict the cumulative future rewards for each action
    (R/W) based on the inputs, which are the last decoder outputs from the
    Transformer model. (1*512 tensor for each sentence).
    """

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors

        input_holder:   Tensor of hidden states
        reward_holder:  Rewards for actions, flatten [max_length, batch_size]
        actions_holder: Chosen actions, flatten [max_length, batch_size]
        dropout_holder: Dropout value for regularisation
        """
        self.input_holder = tf.placeholder(tf.float32, [None, self.config.d_model],
                                           name="input_holder")
        self.reward_holder = tf.placeholder(tf.float32, [None])
        self.actions_holder = tf.placeholder(tf.int32, [None])
        self.dropout_holder = tf.placeholder(tf.float32, name="dropout_holder")

    def create_feed_dict(self, inputs_batch, actions=None, rewards=None, dropout=1):
        """Creates the feed_dict for the linear RL agent.

        Args:
            inputs_batch:   A batch of input data.
            actions:        A batch of chosen actions [max_length*batch_size]
            rewards:        Rewards for chosen actions [max_length*batch_size]
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
            feed_dict[self.reward_holder] = rewards
        return feed_dict

    def add_prediction_op(self):
        """Adds the rewards prediction NN with a single hidden layer
            h = Relu(x*W + b1)
            h_drop = Dropout(h, dropout_rate)
            pred = h_drop*U + b2

        The high reward actions are chosen to expand the translation hypotheses,
        which will update the last hidden states x and hence the new predicted
        cumulative rewards.

        The predicted probabilities for the chosen actions at each time step are
        recorded.

        The true rewards at each time step are calculated based on the
        translation and the target translation placeholders.

        Returns:
            pred: Predicted probabilities of actions
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
            self.preds = tf.nn.softmax(tf.matmul(h_drop, U) + b2, name="predictions")

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
        mask = tf.not_equal(self.reward_holder, zero)
        # correct_decisions = tf.greater(self.reward_holder, 0.5)
        # actions_ref = tf.gather(tf.stack([1 - actions, actions]),
        #                         tf.cast(correct_decisions, tf.int32))
        indices = tf.stack(
            [tf.range(tf.shape(probs_history)[0]), self.actions_holder],
            axis=1 ) # which element to pick in 2D array
        chosen_actions = tf.maximum(0.01, tf.gather_nd(probs_history, indices))
        raw_loss = tf.log(chosen_actions) * self.reward_holder
        loss = -tf.reduce_sum(tf.boolean_mask(raw_loss, mask))
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
        for step in range(self.config.max_length-1):# -1 since already have a READ
            #prev_lengths = [len(x[0].actions) for x in self.cur_hypos]
            hidden_states[step,:,:] = self._get_hidden_states()
            actions[step,:], probs, _ = self.predict_one_step(sess,
                                                       hidden_states[step,:,:])
            #logging.info("step %d     actions:" % step)
            #logging.info(actions[step,:])
            #logging.info("probs:")
            #logging.info(np.max(probs, axis=1))
            #logging.info("Actions length of 1st sentence %d" % len(self.cur_hypos[0][0].actions))
            #logging.info("\n")
            self._update_hidden_states(actions[step,:])
            #new_lengths = [len(x[0].actions) for x in self.cur_hypos]
            #logging.info("change in actions length:")
            #logging.info(np.array(new_lengths)-np.array(prev_lengths))
            #logging.info(np.array([x[0].netRead for x in self.cur_hypos]))
            #logging.info("\n")

        batch_average_delay = 0.0
        BLEU_hypos = []
        BLEU_refs = []
        # Generate full hypotheses from partial hypotheses
        for idx, hypo in enumerate(self.cur_hypos):
            self.cur_hypos[idx] = hypo[0].generate_full_hypothesis()
            batch_average_delay += self.cur_hypos[idx].get_average_delay()
            BLEU_hypos.append([str(x) for x in self.cur_hypos[idx].trgt_sentence])
            BLEU_refs.append([[str(x) for x in targets_batch[idx]]])
        cum_rewards = self._get_bacth_cumulative_rewards(targets_batch)

        _, quality = corpus_bleu(BLEU_refs, BLEU_hypos) # BLEU score for the batch
        batch_average_delay /= len(self.cur_hypos)
        logging.info("\n     batch average delay: %f\n" % batch_average_delay)
        logging.info("\n     batch BLEU:          %f\n" % quality)

        train_dict = self.create_feed_dict(
            np.reshape(hidden_states, (-1, self.config.d_model)),
            actions.flatten(),
            cum_rewards.flatten(),
            self.config.dropout )

        return train_dict

    def _get_bacth_cumulative_rewards(self, targets_batch):
        """Calculate the Cumulative rewards for the full hypotheses stored in
        ``self.cur_hypos'', where the rewards encode the delay and
        the quality.

        Args:
            targets_batch:  A batch of target data (correct translation).
        Returns:
            cum_rewards:    Cumulative rewards for all current hypotheses,
                            numpy array of shape [max_length, batch_size]
        """
        cum_rewards = np.zeros((self.config.max_length, len(targets_batch)),
                               dtype=np.float32)
        for i in range(len(targets_batch)):
            #logging.info("Type of hypo is %s" % type(self.cur_hypos[i]))
            cum_rewards[:,i] = self.cur_hypos[i].get_delay_rewards(self.config)
            # find indices of writes in the action list
            indices = [k for k, x in enumerate(self.cur_hypos[i].actions) if x == "w"]
            Rs = [k for k, x in enumerate(self.cur_hypos[i].actions) if x == "r"]
            logging.info("R/length: %d/%d" % (len(Rs), len(self.all_src[self.cur_hypos[i].lst_id])))
            logging.info("R/W: %d/%d" % (len(Rs), len(indices)))
            assert len(Rs) <= len(self.all_src[self.cur_hypos[i].lst_id])

            # check the length of translation hypothesis is the same as number of "w"
            #logging.info("len(indices) = %d" % len(indices))
            #logging.info("len(self.cur_hypos[i].trgt_sentence) = %d" %
            #        len(self.cur_hypos[i].trgt_sentence))
            #logging.info(self.cur_hypos[i].trgt_sentence)
            assert len(indices) <= len(self.cur_hypos[i].trgt_sentence)

            incremental_BLEU, _ = get_incremental_BLEU(
                            self.cur_hypos[i].trgt_sentence, targets_batch[i])
            logging.info(cum_rewards[:,i])
            cum_rewards[indices,i] += incremental_BLEU
            logging.info(cum_rewards[:,i])
        return cum_rewards


    def __init__(self, config):
        super(linearModel, self).__init__(config.args, config.isTargetNet)
        self.config = config
        self.build()
