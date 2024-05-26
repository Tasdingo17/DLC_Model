import typing as tp

import numpy as np

import DLCmodel.states as dlc_states
import DLCmodel.markov_models as mmodels


class DelaysError(Exception):
    pass

class ProbsError(Exception):
    pass


class DLCModel:

    def __init__(self, 
                 delay: float, jitter: float,
                 p_loss: float, mu: float,
                 e_b: float, e_gb: float,
                 beta: float):
        self.delay = delay
        self.jitter = jitter
        self.p_loss = p_loss
        self.mu = mu
        self.e_b = e_b
        self.e_gb = e_gb
        self.beta = beta

        self._check_input_for_delays()
        m1, j1, m2, j2, m3, j3 = self._get_delay_params(delay, jitter)
        #print(f"Delay params: m1={m1}, j1={j1}, m2={m2}, j2={j2}, m3={m3}, j3={j3}")
        self._state_1 = dlc_states.DLCSimpleState(0, m1, j1)
        self._state_2 = dlc_states.DLCQueueState(0, m2, j2, max_delay=delay+jitter)
        self._state_3 = dlc_states.DLCLossState(1, m3)
        start_distribution = np.array([1.0, 0.0, 0.0])
        self._check_input_for_probs()
        transition_probs = self._get_transition_probs()
        #print("Transition probs:\n", transition_probs)
        self._markov_chain = mmodels.StationaryMarkovChain(
            [self._state_1, self._state_2, self._state_3],
            transition_probs,
            start_distribution
        )

    def _get_transition_probs(self) -> np.ndarray:
        p32 = 1 / self.e_b
        t = (1 / (1 - self.p_loss) - 1) / (self.e_b * (1-self.mu))
        p23 = ((1-self.mu)/ self.mu) * t
        p21 = 1 / self.e_gb - p23
        p12 = self.mu/((1-self.mu) * self.e_gb) - t
        return np.array(
            [[1 - p12, p12, 0.0], 
             [p21, 1 - p21 - p23, p23],
             [0.0, p32, 1 - p32]]
        )

    def _get_delay_params(self, mean_delay: float, jitter: float) -> tp.Tuple[float]:
        """Return m1, j1, m2, j2, m3, j3"""
        m3 = mean_delay + jitter
        j3 = 0
        j1 = jitter * self.beta
        m1 = mean_delay - jitter + j1
        pi3 = self.p_loss
        pi2 = self.mu * (1 - self.p_loss)
        pi1 = (1 - self.p_loss) * (1 - self.mu)
        #print(f"pi1={pi1}, pi2={pi2}, pi3={pi3}")
        m2 = (mean_delay - pi1 * m1 - pi3 * m3) / pi2
        j2 = max(m3 - m2, m2 - (m1+j1))
        return m1, j1, m2, j2, m3, j3


    def _check_input_for_probs(self):
        if not (self.mu * self.e_b - self.e_gb * (self.p_loss / (1 - self.p_loss)) > 0):
            raise ProbsError()


    def _check_input_for_delays(self):
        beta_threshold = self.get_beta_threshold(self.p_loss, self.mu)
        #print("Beta threshold is", beta_threshold)
        if not self.beta > beta_threshold:
            raise DelaysError()

    @staticmethod
    def get_beta_threshold(p_loss, mu):
        pi1 = (1 - p_loss) * (1 - mu)
        beta_threshold = 2 - 1 / pi1
        return beta_threshold

    def gen_sequence(
            self,
            n_samples: int, 
            ret_states: bool = False
    ) -> tp.Union[tp.List, tp.Tuple[tp.List, tp.List[int]]]:
        if not ret_states:
            return [self._markov_chain.step() for _ in range(n_samples)]
        res_packets = [()] * n_samples
        res_states = [0] * n_samples 
        for i in range(n_samples):
            packet, state = self._markov_chain.step(True)
            res_packets[i] = packet
            res_states[i] = state
        return res_packets, res_states
