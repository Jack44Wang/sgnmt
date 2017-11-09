"""``SimHypothesis`` and ``SimPartialHypothesis`` implementation for
simultaneous translation.
"""

import copy

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

    def get_average_delay(self):
        """Return the average delay based on the set of actions taken.
        AP in Gu's paper
        """
        current_delay = 0
        cum_delay = 0
        for action in self.actions:
            if action == 'R':
                current_delay += 1
            else:
                cum_delay += current_delay
        return cum_delay / (current_delay*(len(self.actions)-current_delay))

    def get_consecutive_wait(self):
        """Return the longest wait length (longest consecutive READs)
        based on the set of actions taken.
        CW in Gu's paper
        """
        max_delay = 0
        current_delay = 0
        for action in self.actions:
            if action == 'R':
                current_delay += 1
                if current_delay > max_delay:
                    max_delay = current_delay
            else:
                current_delay = 0
        return max_delay

class SimPartialHypothesis(PartialHypothesis):
    """Represents a partial hypothesis in simultaneous translation. """

    def __init__(self, initial_states = None):
        """Creates a new partial hypothesis with zero score and empty
        translation prefix.

        Args:
            initial_states: Initial predictor states
        """
        super(SimPartialHypothesis, self).__init__(initial_states)
        self.actions = []

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
        hypo = super(SimPartialHypothesis, self).expand(word, score,
                                                        score_breakdown)
        # expanding the hypothesis so the action is WRITE
        hypo.actions = self.actions + ['w']
        return hypo

    def cheap_expand(self, word, score, score_breakdown):
        """Call parent method ``cheap_expand()`` and append WRITE action
        """
        hypo = super(SimPartialHypothesis, self).cheap_expand(word, score,
                                                              score_breakdown)
        # expanding the hypothesis so the action is WRITE
        hypo.actions = self.actions + ['w']
        return hypo