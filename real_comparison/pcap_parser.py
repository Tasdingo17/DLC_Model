import sys

import numpy as np
import scipy.stats as sps
import pandas as pd
import matplotlib.pyplot as plt

from DLCmodel import stats as dlc_stats


#tshark -r ../iperf3_sas2msk.pcapng -Y "ipv6.src eq 2a02:6b8:b081:b50a::1:27 and ipv6.dst eq 2a02:6b8:c1b:221f:0:4438:b83a:0 and tcp.srcport eq 62694 and tcp.dstport eq 5201" -e frame.number -e tcp.seq -e frame.time_epoch -e tcp.analysis.retransmission -e tcp.analysis.out_of_order -E separator=, -T fields > sas2msk.txt
#remove packect with seq=1

def transform_delays(delays: list[float]) -> list[float]:
    res = [0] * len(delays)
    delay_max = max(delays) * 100 + 0.05
    for i, delay in enumerate(delays):
        if delay != -1:
            res[i] = delay * 100
        else:
            res[i] = delay_max
    return res


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

    plt.savefig("real_combined.svg", format='svg')


if __name__ == '__main__':
    filename_src, filename_dst = sys.argv[1], sys.argv[2]
    losses, delays = [], []
    df_src, df_dst = pd.read_csv(filename_src), pd.read_csv(filename_dst)
    lost = set()
    for i, src_row in df_src.iterrows():
        seq = int(src_row['seq'])

        src_rows = df_src.loc[df_src['seq'] == seq]
        if len(src_rows) > 2:
            continue    # lost retransmission
        dst_rows: pd.DataFrame = df_dst.loc[df_dst['seq'] == seq]
        if len(dst_rows) != 1:
            continue    # lost ack, or other hard case
        #if len(dst_row) != 1:
        #    print("len(dst_row) != 1, seq:", seq)
        #    raise RuntimeError()
        dst_row = dst_rows.iloc[0]

        if dst_row['retransmission'] != 1 and dst_row['out_of_order'] != 1:
            if src_row['retransmission'] == 1 or src_row['out_of_order'] == 1:  # lost burst retransmission
                losses.append(1)
                delays.append(-1)
                continue
            losses.append(0)
            delay = dst_row['time_epoch'] - src_row['time_epoch']
            delays.append(delay)
        elif src_row['retransmission'] == 1 or src_row['out_of_order'] == 1:  # recieved retransmission or out_of_order
            if seq not in lost:
                print("seq not in lost, seq:", seq)
                raise RuntimeError()
            lost.remove(seq)
            losses.append(0)
            delay = dst_row['time_epoch'] - src_row['time_epoch']
            if delay < 0:
                breakpoint()
            delays.append(delay)
        else:
            lost.add(seq)
            losses.append(1)
            delays.append(-1) # will be changed to max

    print('unfound lost:', lost)
    delays = transform_delays(delays)

    #draw(losses, delays)

    print("average loss:", dlc_stats.get_average_loss(losses))
    print("average delay:", dlc_stats.get_average_delay(delays))
    print("jitter:", dlc_stats.get_jitter(delays))
    print("average loss burst len:", dlc_stats.get_average_loss_burst_len(losses))

    states = dlc_stats.get_states(losses, delays, 0.55)
    print("mu:", dlc_stats.get_mu_by_states(states))
    print("average good burst len:", dlc_stats.get_average_good_burst_len_by_states(states))

    corr = sps.pearsonr(delays, losses)[0]
    corr = corr if corr is not np.nan else 0.0
    print("Pearson corr:", corr)
