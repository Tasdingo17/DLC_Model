import sys
import csv
import pickle
from time import sleep

import numpy as np
import scipy.stats as sps


import DLCmodel
import DLCmodel.stats as dlc_stats


#MU = 0.6    # mu = pi2 / (pi1 + pi2), i.e. queued_packes / total_arrived_packets
#P_loss = 0.0001   # P_loss = pi3
#E_b = 3
#E_gb = 30

#BETA = 0.25 # jitter1 = jitter * beta

#DELAY = 15
#JITTER = 3

#####################

MU_ARR = [0.2, 0.4, 0.6, 0.8]
P_LOSS_ARR = [0.0001, 0.001, 0.01, 0.05]
E_B_ARR = [2.5, 5, 7]
E_GB_ARR = [20, 30, 40]

BETA = 0.2

DELAY_ARR = [10, 25, 50, 100]
JITTER_ARR = [1, 3, 5, 10]

N_PACKETS = 500000
N_REPEATS = 5


if __name__ == '__main__':
    processed = set()
    with open(sys.argv[2], 'rb') as processed_params_file:
        processed = pickle.load(processed_params_file)

    res_file_name = sys.argv[1]
    HEADER = ['i', 'is_skipped', 'mu', 'est_mu', 'p_loss', 'est_p_loss', 'e_b', 'est_e_b', 'e_gb', 'est_e_gb', 'delay', 'est_delay', 'jitter', 'est_jitter', 'beta', 'correlation']
    with open(res_file_name, 'w') as exps_file:
        writer_total = csv.writer(exps_file)
        writer_total.writerow(HEADER)
        for mu in MU_ARR:
            for p_loss in P_LOSS_ARR:
                for e_b in E_B_ARR:
                    for e_gb in E_GB_ARR:
                        for delay in DELAY_ARR:
                            for jitter in JITTER_ARR:
                                if (mu, p_loss, e_b, e_gb, delay, jitter) in processed:
                                    continue

                                tmp_beta = DLCmodel.DLCModel.get_beta_threshold(p_loss, mu) + 0.01
                                if BETA > tmp_beta:
                                    tmp_beta = BETA
                                print(f"mu={mu}, p_loss={p_loss}, e_b={e_b}, e_gb={e_gb}, delay={delay}, jitter={jitter}, beta={tmp_beta}")
                                for i in range(N_REPEATS):
                                    try:
                                        dlc_model = DLCmodel.DLCModel(delay, jitter, p_loss, mu, e_b, e_gb, tmp_beta)
                                    except DLCmodel.ProbsError:
                                        writer_total.writerow([i, True, mu, -1, p_loss, -1, e_b, -1, e_gb, -1, delay, -1, jitter, -1, tmp_beta, -1])
                                        continue
                                    packets, real_states = dlc_model.gen_sequence(N_PACKETS, True)
                                    sleep(0.1)
                                    losses = [packet[0] for packet in packets]
                                    delays = [packet[1] for packet in packets]
                                    est_mu =  dlc_stats.get_mu_by_states(real_states)
                                    est_loss = dlc_stats.get_average_loss(losses)
                                    est_e_b = dlc_stats.get_average_loss_burst_len(losses)
                                    est_e_gb = dlc_stats.get_average_good_burst_len_by_states(real_states)
                                    est_delay = dlc_stats.get_average_delay(delays)
                                    est_jitter = dlc_stats.get_jitter(delays)
                                    corr = sps.pearsonr(delays, losses)[0]
                                    corr = corr if corr is not np.nan else 0.0
                                    writer_total.writerow([i, False, mu, est_mu, p_loss, est_loss, e_b, est_e_b, e_gb, est_e_gb, delay, est_delay, jitter, est_jitter, tmp_beta, corr])
                                exps_file.flush()
                                sleep(1)
                                