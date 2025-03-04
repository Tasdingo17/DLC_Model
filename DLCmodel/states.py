import abc
import numbers
import typing as tp

import numpy as np
import scipy.stats as sps


MAX_QUEUE_SMOOTH_SAMPLES = 5


class StateBase(abc.ABC):

    @abc.abstractmethod
    def do_work(self) -> tp.Any:
        raise NotImplementedError()
    

class DummyState(StateBase):

    def do_work(self):
        return 1
    

class DummyStateConst(StateBase):

    def __init__(self, delay_val: numbers.Number):
        self._delay_val = delay_val

    def do_work(self):
        return self._delay_val
    

class DummyStateDoubleConst(StateBase):

    def __init__(self, delay_val: numbers.Number, loss_val: numbers.Number):
        self._loss_val = loss_val
        self._delay_val = delay_val

    def do_work(self) -> tp.Tuple[numbers.Number, numbers.Number]:
        return (self._delay_val, self._loss_val)


class OneDistributionState(StateBase):

    def __init__(self, prob_distr) -> None:
        super().__init__()
        self._prob_distr = prob_distr

    def do_work(self) -> numbers.Number:
        return self._prob_distr.rvs(size=1)[0]


class TwoDistributionState(StateBase):

    def __init__(self, loss_prob: float, delay_distr) -> None:
        super().__init__()
        self._loss_distr = sps.bernoulli(loss_prob)
        self._delay_distr = delay_distr

    def do_work(self) -> tp.Tuple[int, numbers.Number]:
        loss_val = self._loss_distr.rvs(size=1)[0]
        delay_val = self._delay_distr.rvs(size=1)[0]
        return (loss_val, delay_val)
    
#################################################################

class DLCSimpleState(StateBase):

    def __init__(self, loss_prob: float, delay_mean: float, jitter: float) -> None:
        super().__init__()
        self._loss_prob = loss_prob # not used
        self._delay_mean = delay_mean
        self._jitter = jitter
        self._jitter_sqrt = jitter**(0.5)

    def do_work(self) -> tp.Tuple[int, numbers.Number]:
        loss_val = 0
        delay_val = self._get_delay_bounded()
        return (loss_val, delay_val)
    
    def _get_delay_bounded(self) -> float:
        res = np.random.normal(self._delay_mean, self._jitter_sqrt)
        # error unsafe, but anyway
        EPS = 0.05
        if res > self._delay_mean + self._jitter:
            res = (self._delay_mean + self._jitter) * (1 - EPS)
        if res < self._delay_mean - self._jitter:
            res = (self._delay_mean - self._jitter) * (1 + EPS)
        return res



class DLCQueueState(StateBase):

    def __init__(self, loss_prob: float, delay_mean: float, jitter: float, max_delay: float) -> None:
        super().__init__()
        self._loss_prob = loss_prob     # not used
        self._delay_mean = delay_mean
        self._jitter = jitter
        self._jitter_sqrt = jitter**(0.5)
        self._max_delay = max_delay
        self._max_queue_smooth_samples = MAX_QUEUE_SMOOTH_SAMPLES
        self._delay_engine = self._random_norm_engine()

    def _random_norm_engine(self):
        start = self._get_delay_bounded()
        stop = self._get_delay_bounded()
        while True:
            yield from self._get_delays_array(start, stop)
            start = stop
            stop = self._get_delay_bounded()

    def _get_delays_array(self, start, stop) -> np.ndarray[float]:
        return np.linspace(
            start,
            stop,
            self._max_queue_smooth_samples, #np.random.randint(1, self._max_queue_smooth_samples + 1),
            endpoint=False,
        )
    
    def _get_delay_bounded(self) -> float:
        res = np.random.normal(self._delay_mean, self._jitter_sqrt)
        # error unsafe, but anyway
        EPS = 0.05
        if res > self._max_delay:
            res = (self._max_delay) * (1 - EPS)
        if res < self._delay_mean - self._jitter:
            res = (self._delay_mean - self._jitter) * (1 + EPS)
        return res

    def do_work(self) -> tp.Tuple[int, numbers.Number]:
        return (0, next(self._delay_engine))


class DLCLossState(StateBase):

    def __init__(self, loss_prob: float, max_delay: float) -> None:
        super().__init__()
        self._loss_prob = loss_prob     # not used
        self._max_delay = max_delay

    def do_work(self) -> tp.Tuple[int, numbers.Number]:
        return (1, self._max_delay)


from DLCmodel.internal_chains import mm1k_delay
class DLCQueueStateV2(StateBase):

    def __init__(self, pi1: float, pi2: float, pi3: float, delay: float, jitter: float, jitter_steps: int) -> None:
        super().__init__()
        self.mm1k = mm1k_delay.MM1K_Delay(pi1, pi2, pi3, delay, jitter, jitter_steps)


    def do_work(self) -> tp.Tuple[int, numbers.Number]:
        return (0, self.mm1k.get_delay())


class DLCLossStateV2(StateBase):

    def __init__(self, max_delay: float) -> None:
        super().__init__()
        self.delay = max_delay

    def do_work(self) -> tp.Tuple[int, numbers.Number]:
        return (1, self.delay)

