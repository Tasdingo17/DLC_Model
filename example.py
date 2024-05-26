from collections import Counter

import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt

import DLCmodel
import DLCmodel.stats as dlc_stats


MU = 0.4    # mu = pi2 / (pi1 + pi2), i.e. queued_packes / total_arrived_packets
P_loss = 0.01   # P_loss = pi3
E_b = 2.5
E_gb = 20

BETA = 0.4 # jitter1 = jitter * beta

DELAY = 15
JITTER = 3

N_PACKETS = 5000


def draw(losses: list[int], delays: list[int]):
    idxs = range(len(losses))

    #задаем местоположение поля с графиком
    plt.figure(figsize=(16,8))
    # заголовок для текущей области 
    plt.title(f'Combined (loss scaled)')
    # рисуем график в текцщей области
    maxdel = max(delays) + 1 
    plt.plot(idxs, [loss * maxdel for loss in losses], label='Loss')
    plt.plot(idxs, delays, label='Delay')
    # сетка графика
    plt.grid(True)
    plt.legend()

    plt.ylabel("Значение")
    plt.xlabel("Номер пакета")

    plt.savefig("combined.svg", format='svg')


def compare_states(true_states: list[int], computed_states: list[int], delays: list[float]) -> bool:
    assert len(true_states) == len(computed_states)
    error_cnt = 0
    for i in range(len(true_states)):
        if true_states[i] != computed_states[i]:
            error_cnt += 1
            #print(f"Error in packet {i}: true_state={true_states[i]}, computed={computed_states[i]}, delay={delays[i]}")
    print("Total states errors:", error_cnt)
    return error_cnt == 0


if __name__ == '__main__':
    dlc_model = DLCmodel.DLCModel(DELAY, JITTER, P_loss, MU, E_b, E_gb, BETA)
    packets, real_states = dlc_model.gen_sequence(N_PACKETS, True)
    #print(counter)
    real_states_counter = Counter(real_states)
    #print(real_states_counter)
    losses = [packet[0] for packet in packets]
    delays = [packet[1] for packet in packets]
    corr = sps.pearsonr(delays, losses)[0]
    corr = corr if corr is not np.nan else 0.0
    print("Pearson corr:", corr)

    draw(losses, delays)

    print("Average loss:", dlc_stats.get_average_loss(losses))
    print("Average loss burst len:", dlc_stats.get_average_loss_burst_len(losses))
    print("Average delay:", dlc_stats.get_average_delay(delays))
    print(f"Min delay={min(delays)}, max delay={max(delays)}")
    print("Jitter:", dlc_stats.get_jitter(delays))
    print("MU:", dlc_stats.get_mu_by_states(real_states))
    
    computed_states = dlc_stats.get_states(losses, delays, BETA)
    print("Average good burst len by real:", dlc_stats.get_average_good_burst_len_by_states(real_states))
    print("Average good burst len by computed:", dlc_stats.get_average_good_burst_len_by_states(computed_states))
    #compare_states(real_states, computed_states, delays)
