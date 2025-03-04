import abc
import typing as tp

import numpy as np

from DLCmodel import states


class MarkovChainBase(abc.ABC):

    def __init__(self, 
                 states: tp.List[states.StateBase], 
                 init_probs: np.ndarray,
                 start_distribution: np.ndarray) -> None:
        self._check_input(states, init_probs, start_distribution)
        self._states = states
        self._probs_orig = init_probs
        self._cumprobs = np.cumsum(init_probs, axis=1)
        self._start_distr_orig = start_distribution
        self._cumstart_distr = np.cumsum(start_distribution)
        self.curr_state = self._get_init_state()

    @staticmethod
    def _check_input(states: tp.List[states.StateBase], 
                     probs: np.ndarray, 
                     start_distribution: np.ndarray) -> None:
        # print(len(states), start_distribution)
        assert len(probs.shape) == 2, probs.shape
        assert probs.shape[0] == probs.shape[1]
        assert probs.shape[0] == len(states)
        assert len(start_distribution.shape) == 1
        assert start_distribution.shape[0] == len(states)

        #assert np.all(probs.sum(axis=1) == 1), probs
        assert not np.any((probs < 0)), probs
        assert start_distribution.sum() == 1, start_distribution

    def _get_init_state(self) -> int:
        # return list(sps.multinomial.rvs(1, self._start_distr_orig)).index(1)
        p = np.random.uniform(0, 1)
        return np.digitize(p, self._cumstart_distr, True)

    def _update_state(self):
        p_tr = self._cumprobs[self.curr_state]
        # self.curr_state = list(sps.multinomial.rvs(1, self._probs_orig[self.curr_state])).index(1)
        p = np.random.uniform(0, 1)
        self.curr_state = np.digitize(p, p_tr, True)

    def step(self, return_state: bool = False) -> tp.Union[tp.Any, tp.Tuple[tp.Any, int]]:
        res = self._states[self.curr_state].do_work()
        state = self.curr_state
        self._update_state()
        self._update_transition_probs()
        if return_state:
            return res, state + 1
        return res

    @abc.abstractmethod
    def _update_transition_probs(self):
        raise NotImplementedError
    

class StationaryMarkovChain(MarkovChainBase):

    def _update_transition_probs(self):
        return