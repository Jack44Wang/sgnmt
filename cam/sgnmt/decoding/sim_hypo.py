"""``SimHypothesis`` and ``SimPartialHypothesis`` implementation for
simultaneous translation.
"""

import copy
import logging
from cam.sgnmt.decoding.core import Hypothesis, PartialHypothesis

class SimHypothesis(Hypothesis):
    """Complete translation hypotheses are represented by an instance
    of this class. Everything included in ``Hypothesis`` class and a
    history of actions taken.
    """

    def __init__(self, trgt_sentence, total_score,
                 score_breakdown = [], actions = []):
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
        self.actions.pop() # remove the last 'w'

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

class SimPartialHypothesis(PartialHypothesis):
    """Represents a partial hypothesis in simultaneous translation. """

    def __init__(self, initial_states = None, max_len = 50, lst_id = -1):
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
                             self.score_breakdown, self.actions)

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
        hypo.netRead -= 1
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
        hypo.netRead -= 1
        return hypo
