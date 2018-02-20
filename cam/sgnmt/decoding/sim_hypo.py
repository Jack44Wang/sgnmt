"""``SimHypothesis`` and ``SimPartialHypothesis`` implementation for
simultaneous translation.
"""

import copy
import logging
import numpy as np
from cam.sgnmt.decoding.core import Hypothesis, PartialHypothesis

class SimHypothesis(Hypothesis):
    """Complete translation hypotheses are represented by an instance
    of this class. Everything included in ``Hypothesis`` class and a
    history of actions taken.
    """

    def __init__(self, trgt_sentence, total_score,
                 score_breakdown = [], actions = [], lst_id = -1):
        """Creates a new full hypothesis for simultaneous translation

        Args:
            trgt_sentence (list): List of target word ids without <S>
                                  or </S> which make up the target sentence
            total_score (float): combined total score of this hypo
            score_breakdown (list): Predictor score breakdown for each
                                    target token in ``trgt_sentence``
            actions (list): List of actions taken during simultaneous
                            translation
        """
        super(SimHypothesis, self).__init__(trgt_sentence,
                                            total_score,
                                            score_breakdown)
        self.actions = actions
        self.lst_id = lst_id
        if self.actions[-1] == 'w' and trgt_sentence[-1] < 4: 
            # reserved words, not in translation
            self.actions.pop() # remove the last 'w'
            self.trgt_sentence.pop()

    def get_average_delay(self):
        """Return the average delay based on the set of actions taken.
        AP in Gu's paper
        """
        current_delay = 0
        cum_delay = 0
        logging.info(self.actions)
        for action in self.actions:
            if action == 'r':
                current_delay += 1
            else:
                cum_delay += current_delay
        if current_delay == 0 or current_delay == len(self.actions):
            return 1.0
        return 1.0*cum_delay / (current_delay*(len(self.actions)-current_delay))

    def get_consecutive_wait(self):
        """Return the longest wait length (longest consecutive READs)
        based on the set of actions taken.
        CW in Gu's paper
        """
        max_delay = 0
        current_delay = 0
        for action in self.actions:
            if action == 'r':
                current_delay += 1
                if current_delay > max_delay:
                    max_delay = current_delay
            else:
                current_delay = 0
        return max_delay

    def get_delay_rewards(self, config):
        """Return the delay rewards (-ve) for each action taken.
        Args:
            config: Configuration object for rewards evaluation
        Returns:
            Rd:     List of delay rewards, size of max_length
        """
        current_consec = 0  # consecutive waits
        current_delay = 0   # number of READs so far
        cum_delay = 0       # cumulative delays

        Rd = np.zeros(config.max_length)
        #logging.info("actions length: %d" % len(self.actions))
        logging.info(self.actions)

        prev_dt = config.d_trg
        for i, action in enumerate(self.actions):
            if action == 'r':
                current_consec += 1
                current_delay += 1
                consec_penalty = 2 if current_consec > config.c_trg else 0
            else:
                current_consec = 0
                consec_penalty = 0
                cum_delay += current_delay
            if cum_delay == 0:
                dt = 1.0 # all reads
            else:
                dt = 1.0*cum_delay / (current_delay*(i - current_delay + 1))
            ap_penalty = (dt - prev_dt) if dt > config.d_trg else 0
            #logging.info("%d/%d" % (i, config.max_length))
            prev_dt = dt
            Rd[i] = config.alpha*consec_penalty + config.beta*ap_penalty
        #logging.info("Delay rewards Rd:")
        #Rd[0] = 0.0 # no penalty for the first read
        logging.info(Rd)

        return Rd

    def get_last_delay_reward(self, config):
        """Return the delay reward (-ve) for the hypothesis
        Args:
            config: Configuration object for rewards evaluation
        Returns:
            Rd:     Final delay reward
        """
        logging.info(self.actions)
        consec_penalty = 2 if self.get_consecutive_wait() > config.c_trg else 0
        dt = self.get_average_delay()
        ap_penalty = (dt - config.d_trg) if dt > config.d_trg else 0

        Rd = config.alpha*consec_penalty + config.beta*ap_penalty
        logging.info(Rd)

        return Rd

class SimPartialHypothesis(PartialHypothesis):
    """Represents a partial hypothesis in simultaneous translation. """

    def __init__(self, initial_states = None, max_len = 60, lst_id = -1):
        """Creates a new partial hypothesis with zero score and empty
        translation prefix.

        Args:
            initial_states: Initial predictor states
            max_len:        Maximum number of decoding iterations (R+W)
            lst_id:         The index of sentence in ``model.all_src''
        """
        super(SimPartialHypothesis, self).__init__(initial_states)
        self.actions = ['r']
        self.progress = 1;
        self.netRead = 1 # number of R - number of W
        self.max_len = max_len
        self.lst_id = lst_id

    def generate_full_hypothesis(self):
        """Create a ``SimHypothesis`` instance from this hypothesis. """
        return SimHypothesis(self.trgt_sentence, self.score,
                         self.score_breakdown, self.actions, self.lst_id)

    def append_action(self, action = 'r'):
        """Append a new action to the list of actions. """
        self.actions += [action]

    def expand(self, word, new_states, score, score_breakdown):
        """Call parent method ``expand()`` and append WRITE action
        """
        hypo = SimPartialHypothesis(new_states, self.max_len, self.lst_id)
        hypo.score = self.score + score
        hypo.score_breakdown = copy.copy(self.score_breakdown)
        hypo.trgt_sentence = self.trgt_sentence + [word]
        hypo.add_score_breakdown(score_breakdown)
        # expanding the hypothesis so the action is WRITE
        hypo.actions = self.actions + ['w']
        hypo.progress = self.progress
        hypo.netRead = self.netRead - 1
        return hypo

    def cheap_expand(self, word, score, score_breakdown):
        """Call parent method ``cheap_expand()`` and append WRITE action
        """
        hypo = SimPartialHypothesis(self.predictor_states,
                                    self.max_len, self.lst_id)
        hypo.score = self.score + score
        hypo.score_breakdown = copy.copy(self.score_breakdown)
        hypo.trgt_sentence = self.trgt_sentence + [word]
        hypo.word_to_consume = word
        hypo.add_score_breakdown(score_breakdown)
        # expanding the hypothesis so the action is WRITE
        hypo.actions = self.actions + ['w']
        hypo.progress = self.progress
        hypo.netRead = self.netRead - 1
        return hypo
