import typing as tp
from collections import Counter

import numpy as np

def get_average_loss(losses: list[int]) -> float:
    return np.array(losses).mean()


def get_average_delay(delays: list[float]) -> float:
    return np.array(delays).mean()


def get_jitter(delays: list[float], mean: tp.Optional[float] = None) -> float:
    true_mean = mean
    tmp = np.array(delays)
    if not mean:
        true_mean = tmp.mean()
    return max(true_mean - tmp.min(), tmp.max() - true_mean)


def get_average_loss_burst_len(losses: list[int]) -> float:
    burst_lens = []
    start, stop = 0, 0
    while start < len(losses):
        if losses[start] == 1:
            stop = start
            while stop < len(losses) and losses[stop] == 1:
                stop += 1
            burst_lens.append(stop - start)
            start = stop
        start += 1
    if not burst_lens:
        return 0
    return np.array(burst_lens).mean()


def get_average_good_burst_len_by_states(states: list[int]) -> float:
    burst_lens = []
    start, stop = 0, 0
    while start < len(states):
        if states[start] == 2:
            stop = start
            while stop < len(states) and states[stop] == 2:
                stop += 1
            burst_lens.append(stop - start)
            start = stop
        start += 1
    if not burst_lens:
        return 0
    return np.array(burst_lens).mean()


def get_mu_by_states(states: list[int]) -> float:
    counter = Counter(states)
    if not counter.get(2):
        return 0
    return counter[2] / (counter[1] + counter[2])


def get_mu_by_packets(losses: list[int], delays: list[int]) -> float:
    states = get_states(losses, delays)
    return get_mu_by_states(states)


def get_states(losses: list[int], delays: list[float], beta: float) -> list[int]:
    assert len(losses) == len(delays)
    queue_threshold = _get_queue_threshold(delays, beta)
    res = [0] * len(losses)     # reserve 
    for i in range(len(losses)):
        if losses[i] == 1:
            res[i] = 3
            continue
        if delays[i] > queue_threshold:
            res[i] = 2
        else:
            res[i] = 1
    return res


def _get_queue_threshold(delays: list[float], beta: float) -> float:
    """
    Threshold is (m1 + j1).
    j1 = jitter * beta
    m1 = mean - jitter + j1
    threshold = mean + jitter(2*beta - 1)
    """
    mean = get_average_delay(delays)
    jitter = get_jitter(delays, mean)
    return mean + jitter * (2 * beta - 1)
