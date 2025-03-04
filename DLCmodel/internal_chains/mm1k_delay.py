import numpy as np
import typing as tp

from DLCmodel import markov_models
from DLCmodel import states

EPS_PROB = 0.0001


def newton_method(func: tp.Callable, derivative: tp.Callable, start: float = 0.5) -> float:
    x = start
    x_prev = -1
    while(abs(x - x_prev) > EPS_PROB):
        x_prev = x
        x = x - func(x) / derivative(x)
        # print(f'{x_prev} -> {x}')
    return x


class MM1K_Delay:
    def __init__(self, pi1: float, pi2: float, pi3: float, delay: float, jitter: float, jitter_steps: int):
        self._d = delay
        self._calc_d2(pi1, pi2, delay, jitter, jitter_steps)
        print(f'pi1={pi1}, pi2={pi2}, pi3={pi3}, n_steps={self._j_steps}, step={self._step}, d2={self._d2}, offset={self._offset}')

        states_list = [states.DummyStateConst(delay + i * self._step) for i in range(self._j_steps+1)]
        start_distr = np.array([1] + [0] * self._j_steps)
        init_probs = self._construct_mm1_probs()
        self.MM1_model = markov_models.StationaryMarkovChain(states_list, init_probs, start_distr)
    
    def _calc_d2(self, pi1: float, pi2: float, delay: float, jitter: float, jitter_steps: int) -> None:
        j_steps = jitter_steps
        step = jitter / j_steps
        d2 = (delay - pi1 * (delay - step)) / pi2
        while d2 >= (delay + jitter):
            j_steps += 1
            step = jitter / j_steps
            d2 = (delay - pi1 * (delay - step)) / pi2
        if j_steps != jitter_steps:
            print(f"Increasdc j_steps to agjust d2, j_steps={j_steps}")

        self._j_steps = j_steps
        self._step = step
        self._d2 = d2
        self._offset = (self._d2 - delay) / self._step
        return

    def _construct_mm1_probs(self) -> np.ndarray:
        def _func(x: float) -> float:
            x_pw = pow(x, self._j_steps+1)
            return x/(1-x) - (self._j_steps+1) * x_pw/(1 - x_pw) - self._offset

        def _derivative(x: float) -> float:
            return 1/(1-x) + x/(1-x)**2 - (self._j_steps+1)**2 * pow(x, self._j_steps)/(1 - pow(x, self._j_steps+1)) - (self._j_steps+1)**2 * pow(x, 2*self._j_steps + 1)/(1 - pow(x, self._j_steps+1))**2

        rho = newton_method(_func, _derivative)
        p_min, p_plus = 1/(1+rho), rho/(1+rho)
        #print(f'rho={rho}, p_min={p_min}, p_plus={p_plus}')

        k_states = self._j_steps+1
        res = np.zeros((k_states, k_states))
        res[0][0], res[0][1] = 1 - p_plus, p_plus     # init first state
        res[-1][-1], res[-1][-2] = 1 - p_min, p_min   # init last state
        for i in range(1, k_states-1):
            res[i][i-1], res[i][i+1] = p_min, p_plus
        return res


    def get_delay(self) -> float:
        return self.MM1_model.step()
