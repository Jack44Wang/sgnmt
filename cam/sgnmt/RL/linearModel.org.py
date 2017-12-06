import logging
from datetime import datetime

import tensorflow as tf
import numpy as np

from model import Model

class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    d_model = 512
    max_length = 40 # longest BPE units in a sentence
    dropout = 0.2
    hidden_size = 64
    batch_size = 2 #32
    n_epochs = 10
    lr = 0.001

    def __init__(self, args):
        self.args = args

        if "model_path" in args:
            # Where to save things.
            self.output_path = args.model_path
        else:
            self.output_path = "results/{:%Y%m%d_%H%M%S}/".format(datetime.now())
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

        input_placeholder:   Tensor of hidden states
        targets_placeholder: Tensor of reference translation
        target_mask_placeholder: Mask for targets
        dropout_placeholder: Dropout value for regularisation
        reward_holder:       Cumulative future rewards for each sentence
        probs_holder:        Probabilities of chosen actions for each sentence
        """
        self.input_placeholder = tf.placeholder(tf.float32, 
                            [None, self.config.d_model])
        self.targets_placeholder = tf.placeholder(tf.float32,
                            [None, self.config.max_length, self.config.d_model])
        self.target_mask_placeholder = tf.placeholder(tf.bool,
                            [None, self.config.max_length])
        self.dropout_placeholder = tf.placeholder(tf.float32)
        self.reward_holder = tf.placeholder(tf.float32, 
                            [None, self.config.max_length])

    def create_feed_dict(self, inputs_batch, mask_batch=None,
                            targets_batch=None, dropout=1):
        """Creates the feed_dict for the linear RL agent.

        Args:
            inputs_batch:  A batch of input data.
            targets_batch: A batch of targets data.
            mask_batch:    A batch of mask data to the targets.
            dropout:       The dropout rate.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        feed_dict = {
            self.input_placeholder: inputs_batch,   
            self.dropout_placeholder: dropout
        }
        if targets_batch is not None:
            feed_dict[self.targets_placeholder] = targets_batch
            feed_dict[self.target_mask_placeholder] = mask_batch
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
            pred: tf.Tensor of shape (batch_size, 2)
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

            h = tf.nn.relu(tf.matmul(self.input_placeholder, W) + b1)
            h_drop = tf.nn.dropout(h, self.dropout_placeholder)
            self.preds = tf.nn.softmax(tf.matmul(h_drop, U) + b2)

        return self.preds

    def add_loss_op(self, probs_history):
        """Adds Ops for the loss function to the computational graph.
        Use the mask to mask out nan produced when evaluating log(0)*0, since
        the last few elements of probs_holder will be 0's for each sentence.
        
        Returns:
            loss: A 0-d tensor (scalar)
        """
        zero = tf.constant(0, dtype=tf.float32)
        mask = tf.not_equal(self.reward_holder, zero)
        #logging.info(mask.shape)
        raw_loss = tf.log(probs_history)*self.reward_holder
        loss = -tf.reduce_sum(tf.boolean_mask(raw_loss, mask))
        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.

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

    def predict_on_batch(self, sess, inputs_batch):
        feed = self.create_feed_dict(inputs_batch)
        predictions = sess.run(tf.argmax(self.pred, axis=2), feed_dict=feed)
        return predictions

    def train_on_batch(self, sess, inputs_batch, targets_batch, mask_batch):
        """Use a for loop to update the 'input_placeholder' repeatedly,
        record the actions chosen for each step. Use these to calculate the
        cumulative rewards for the action taken at each timestep.
        """
        probs_history = np.zeros((inputs_batch.shape[0], self.config.max_length),
                                  dtype=np.float32)
        for step in range(self.config.max_length):
            # get the actions with probabilities and save them
            actions, probs_history[:,step] = self.predict_one_step(sess,
                                                                   inputs_batch)
            self._update_hidden_states(actions)
            inputs_batch = self._get_hidden_states()

        # get the cumulative rewards for the batch
        cum_rewards = self._get_bacth_cumulative_rewards(targets_batch, mask_batch)

        # get loss definition
        self.loss = self.add_loss_op(probs_history)
        self.train_op = self.add_training_op(self.loss)

        # train on batch, returns the loss to monitor
        feed = {self.reward_holder:cum_rewards}
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def predict_one_step(self, sess, inputs_batch):
        """Make the prediction of actions by selecting the most probable ones.

        Returns:
            probs:   Probabilities of chosen actions, [batch_size * 1]
            actions: Chosen actions, [batch_size * 1]
                     0 -> READ
                     1 -> WRITE
        """
        feed = self.create_feed_dict(inputs_batch, dropout=self.config.dropout)
        predictions = sess.run(self.pred, feed_dict=feed)
        probs = sess.run(tf.reduce_max(predictions, axis=1))
        actions = sess.run(tf.argmax(predictions, axis=1))
        # TODO check if able to WRITE (ie written < read)

        return actions, probs

    def __init__(self, config):
        super(linearModel, self).__init__(config.args)
        self.config = config
        self.build()
