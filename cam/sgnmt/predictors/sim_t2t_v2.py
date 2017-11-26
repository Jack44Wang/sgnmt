"""This is the interface to the tensor2tensor library.

https://github.com/tensorflow/tensor2tensor

Alternatively, you may use the following fork which has been tested in
combination with SGNMT:

https://github.com/fstahlberg/tensor2tensor

The t2t predictor can read any model trained with tensor2tensor which
includes the transformer model, convolutional models, and RNN-based
sequence models.
"""

import logging
import os

from cam.sgnmt import utils
from cam.sgnmt.predictors.tf_t2t import expand_input_dims_for_t2t
from cam.sgnmt.predictors.tf_t2t import _BaseTensor2TensorPredictor
from cam.sgnmt.predictors.tf_t2t import log_prob_from_logits


POP = "##POP##"
"""Textual representation of the POP symbol."""

try:
    # Requires tensor2tensor
    from tensor2tensor.utils import trainer_utils
    from tensor2tensor.utils import usr_dir
    from tensor2tensor.utils import registry
    from tensor2tensor.utils import devices
    from tensor2tensor.data_generators import text_encoder
    import tensorflow as tf
    import numpy as np
    from cam.sgnmt.predictors.tf_t2t import DummyTextEncoder
    """
    # Define flags from the t2t binaries
    flags = tf.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string("schedule", "train_and_evaluate",
                        "Method of tf.contrib.learn.Experiment to run.")
    """
except ImportError:
    pass # Deal with it in decode.py


class SimT2TPredictor_v2(_BaseTensor2TensorPredictor):
    """This predictor implements scoring with Tensor2Tensor models. We
    follow the decoder implementation in T2T and do not reuse network
    states in decoding. We rather compute the full forward pass along
    the current history. Therefore, the decoder state is simply the
    the full history of consumed words.
    """

    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 model_name,
                 problem_name,
                 hparams_set_name,
                 t2t_usr_dir,
                 checkpoint_dir,
                 t2t_unk_id=None,
                 single_cpu_thread=False,
                 max_terminal_id=-1,
                 pop_id=-1):
        """Creates a new simultaneous T2T predictor. The constructor prepares
        the TensorFlow session for predict_next() calls. This includes:
        - Load hyper parameters from the given set (hparams)
        - Update registry, load T2T model
        - Create TF placeholders for source sequence and target prefix
        - Create computation graph for computing log probs.
        - Create a MonitoredSession object, which also handles
          restoring checkpoints.

        Args:
            src_vocab_size (int): Source vocabulary size.
            trg_vocab_size (int): Target vocabulary size.
            model_name (string): T2T model name.
            problem_name (string): T2T problem name.
            hparams_set_name (string): T2T hparams set name.
            t2t_usr_dir (string): See --t2t_usr_dir in tensor2tensor.
            checkpoint_dir (string): Path to the T2T checkpoint
                                     directory. The predictor will load
                                     the top most checkpoint in the
                                     `checkpoints` file.
            t2t_unk_id (int): If set, use this ID to get UNK scores. If
                              None, UNK is always scored with -inf.
            single_cpu_thread (bool): If true, prevent tensorflow from
                                      doing multithreading.
            max_terminal_id (int): If positive, maximum terminal ID. Needs to
                be set for syntax-based T2T models.
            pop_id (int): If positive, ID of the POP or closing bracket symbol.
                Needs to be set for syntax-based T2T models.
        """
        super(SimT2TPredictor_v2, self).__init__(t2t_usr_dir,
                                           checkpoint_dir,
                                           t2t_unk_id,
                                           single_cpu_thread)
        self.consumed = []
        self.src_sentence = []
        self.pop_id = pop_id
        self.max_terminal_id = max_terminal_id
        self.previous_encode = -1
        self.previous_decode = -1
        predictor_graph = tf.Graph()
        with predictor_graph.as_default() as g:
            hparams = self._create_hparams(
                src_vocab_size, trg_vocab_size, hparams_set_name, problem_name)
            p_hparams = hparams.problems[0]
            self._inputs_var = tf.placeholder(dtype=tf.int32,
                                              shape=[None],
                                              name="sgnmt_inputs")
            self._targets_var = tf.placeholder(dtype=tf.int32,
                                               shape=[None],
                                               name="sgnmt_targets")
            features = {"problem_choice": tf.constant(0),
                        "input_space_id": tf.constant(p_hparams.input_space_id),
                        "target_space_id": tf.constant(
                            p_hparams.target_space_id),
                        "inputs": expand_input_dims_for_t2t(self._inputs_var),
                        "targets": expand_input_dims_for_t2t(self._targets_var)}

            model = registry.model(model_name)(
                hparams,
                tf.estimator.ModeKeys.PREDICT,
                hparams.problems[0],
                0,
                devices.data_parallelism(),
                devices.ps_devices(all_workers=True))
            sharded_logits, _ = model.model_fn(features)
            self._log_probs = log_prob_from_logits(sharded_logits[0])
            self._encoder_output = model.encoder_output
            self._encoder_decoder_attention_bias = model.attention_bias
            self._decoder_output = model.decoder_output

            self.mon_sess = self.create_session()

    def _create_hparams(
          self, src_vocab_size, trg_vocab_size, hparams_set_name, problem_name):
        """Creates hparams object.

        This method corresponds to create_hparams() in tensor2tensor's
        trainer_utils module, but replaces the feature encoders with
        DummyFeatureEncoder's.

        Args:
            src_vocab_size (int): Source vocabulary size.
            trg_vocab_size (int): Target vocabulary size.
            hparams_set_name (string): T2T hparams set name.
            problem_name (string): T2T problem name.

        Returns:
            hparams object.

        Raises:
            LookupError if the problem name is not in the registry or
            uses the old style problem_hparams.
        """
        hparams = registry.hparams(hparams_set_name)()
        problem = registry.problem(problem_name)
        # The following hack is necessary to prevent the problem from creating
        # the default TextEncoders, which would fail due to the lack of a
        # vocabulary file.
        problem._encoders = {
            "inputs": DummyTextEncoder(vocab_size=src_vocab_size),
            "targets": DummyTextEncoder(vocab_size=trg_vocab_size)
        }
        try:
            hparams.add_hparam("max_terminal_id", self.max_terminal_id)
        except:
            if hparams.max_terminal_id != self.max_terminal_id:
                logging.warn("T2T max_terminal_id does not match (%d!=%d)"
                             % (hparams.max_terminal_id, self.max_terminal_id))
        try:
            hparams.add_hparam("closing_bracket_id", self.pop_id)
        except:
            if hparams.closing_bracket_id != self.pop_id:
                logging.warn("T2T closing_bracket_id does not match (%d!=%d)"
                             % (hparams.closing_bracket_id, self.pop_id))
        p_hparams = problem.get_hparams(hparams)
        hparams.problem_instances = [problem]
        hparams.problems = [p_hparams]
        return hparams

    def predict_next(self):
        """Call the T2T model in self.mon_sess."""
        log_probs = self.mon_sess.run(self._log_probs,
            {self._inputs_var: self.src_sentence,
             self._targets_var: self.consumed + [text_encoder.PAD_ID]})
        log_probs_squeezed = log_probs[0, 0, 0, 0, :]
        log_probs_squeezed[text_encoder.PAD_ID] = utils.NEG_INF
        return log_probs_squeezed

    def initialize(self, src_sentence):
        """Set src_sentence to a prefix, reset consumed.
        If src_sentence is a complete sentence, call reveal(text_encoder.EOS_ID)self.previous_encode = 
        """
        self.consumed = []
        self.src_sentence = src_sentence

    def consume(self, word):
        """Append ``word`` to the current history.
        Add the translated sentence
        """
        self.consumed.append(word)

    def reveal(self, word):
        """Reveal a source word to the predictor.
        Remember to add the [text_encoder.EOS_ID] in the end
        """
        self.src_sentence.append(word)

    def get_hidden_state(self):
        """Get the hidden state in the Transformer model.
        The hidden_states is going to be consumed by the RL agent.
        Returned as np array
        """
        encoder_output = self.mon_sess.run(self._encoder_output,
            {self._inputs_var: self.src_sentence,
             self._targets_var: self.consumed + [text_encoder.PAD_ID]})
        bias = self.mon_sess.run(self._encoder_decoder_attention_bias,
            {self._inputs_var: self.src_sentence,
             self._targets_var: self.consumed + [text_encoder.PAD_ID]})
        decoder_output = self.mon_sess.run(self._decoder_output,
            {self._inputs_var: self.src_sentence,
             self._targets_var: self.consumed + [text_encoder.PAD_ID]})
        if self.previous_encode > -1 and encoder_output.shape[1]-1 > self.previous_encode:
            # new source word
            logging.info("encoder_output difference %s." % 
                np.sum(np.absolute(encoder_output[0,self.previous_encode]-\
                self.pre_enc[0,self.previous_encode])))
            logging.info("previous_Enc -1 %s." % self.pre_enc[0,-1,:4])
            logging.info("new_Enc      -2 %s." % encoder_output[0,self.previous_encode,:4])
            logging.info("new_Enc      -1 %s." % encoder_output[0,-1,:4])
            if self.previous_decode > -1:
                # check if the decoder output is changing with the source prefix
                assert self.previous_decode == decoder_output.shape[1]-1
                logging.info("decoder_output difference %s." % 
                    np.sum(np.absolute(decoder_output[0,-1,0]-self.pre_dec[0,-1,0])))

        if self.previous_decode > -1 and decoder_output.shape[1]-1 > self.previous_decode:
            # new target word
            logging.info("decoder_output difference %s." % 
                np.sum(np.absolute(decoder_output[0,self.previous_decode,0]-\
                self.pre_dec[0,self.previous_decode,0])))
            logging.info("previous_Dec -1 %s." % self.pre_dec[0,-1,0,:4])
            logging.info("new_Enc      -2 %s." % decoder_output[0,self.previous_decode,0,:4])
            logging.info("new_Enc      -1 %s." % decoder_output[0,-1,0,:4])
            if self.previous_encode > -1:
                # check if the encoder output is changing with the target prefix
                assert self.previous_encode == encoder_output.shape[1]-1
                assert np.sum(np.absolute(encoder_output[0,-1]-self.pre_enc[0,-1])) < 1e-9
                logging.info("encoder states unchanged.")

        logging.info("encoder_output size %s." % (encoder_output.shape,))
        #logging.info("attention_bias size %s." % (bias.shape,))
        #logging.info(bias)
        logging.info("decoder_output size %s.\n" % (decoder_output.shape,))
        self.previous_encode = encoder_output.shape[1]-1
        self.pre_enc = encoder_output
        self.previous_decode = decoder_output.shape[1]-1
        self.pre_dec = decoder_output
        return encoder_output, bias, decoder_output

    def get_state(self):
        """The predictor state is the complete history."""
        return self.consumed

    def set_state(self, state):
        """The predictor state is the complete history."""
        self.consumed = state

    def reset(self):
        """Empty method. """
        pass

    def is_equal(self, state1, state2):
        """Returns true if the history is the same """
        return state1 == state2
