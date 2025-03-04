import typing as tp
from collections import Counter

import numpy as np

def get_average_loss(losses: list[int]) -> float:
    return np.array(losses).mean()


def get_average_delay(delays: list[float], losses: list[int]) -> float:
    assert len(losses) == len(delays)
    valuable_delays = [delay for i, delay in enumerate(delays) if losses[i] != 1]
    return np.array(valuable_delays).mean()


def get_jitter(delays: list[float], losses: list[int], mean: tp.Optional[float] = None) -> float:
    assert len(losses) == len(delays)
    valuable_delays = np.array([delay for i, delay in enumerate(delays) if losses[i] != 1])
    true_mean = mean
    if not true_mean:
        true_mean = valuable_delays.mean()
    jitters = np.absolute(valuable_delays - true_mean)
    return jitters.max()


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


def get_states(losses: list[int], delays: list[float], jitter_steps: int) -> list[int]:
    assert len(losses) == len(delays)
    queue_threshold = _get_queue_threshold(delays, losses, jitter_steps)
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


def _get_queue_threshold(delays: list[float], losses: list[int], jitter_steps: int) -> float:
    """
    jitter_step = jitter / jitter_steps
    Threshold is mean + jitter_step: delay < (mean + jitter_step) ? 1 : 2
    """
    mean = get_average_delay(delays, losses)
    jitter = get_jitter(delays, losses, mean)
    return mean + jitter / jitter_steps
