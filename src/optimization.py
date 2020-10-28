import numpy as np
import scipy.stats
from eskf_runner import run_eskf
from scipy.optimize import minimize
import time
import functools


def cost_function_NIS(tunables, *args):
    eskf_parameters = tunables[:4]
    eskf_parameters = np.append(eskf_parameters, args[0])
    eskf_parameters = np.abs(eskf_parameters)
    p_std = np.abs(np.repeat(tunables[4:6], [2, 1]))

    P_pred_init_list, x_init, loaded_data, N, offset, use_GNSSaccuracy = args[1:]
    result = run_eskf(eskf_parameters, x_init, P_pred_init_list, loaded_data,
                      p_std, N, offset=offset,
                      use_GNSSaccuracy=use_GNSSaccuracy)
    # delta_x = result[3]

    NIS = result[4]
    NIS_data = NIS[:, 1]
    CI390 = np.array(scipy.stats.chi2.interval(0.9, 3))
    CI395 = np.array(scipy.stats.chi2.interval(0.95, 3))
    inCIpos90 = np.mean((CI390[0] <= NIS_data) * (NIS_data <= CI390[1]))
    inCIpos95 = np.mean((CI395[0] <= NIS_data) * (NIS_data <= CI395[1]))

    # cost = np.mean(np.log(NIS[:, 1])**2)
    inCIpos90_cost = 1-inCIpos90
    inCIpos95_cost = 1-inCIpos95
    mean_deciance_cost = np.mean(np.log(NIS[:, 1])**2)
    cost = inCIpos90_cost + 0.5*inCIpos95_cost + 0.001 * mean_deciance_cost

    with open('optimization.txt', 'a') as file:
        text = (f'eskf_parameters: {tunables[:4]}\n'
                f'gps_parameters: {tunables[4:]}\n'
                f'cost {inCIpos90_cost}, {inCIpos95_cost}, '
                f'{mean_deciance_cost}, '
                f'{cost}\n\n')
        file.write(text)
        print(text)
        time.sleep(0.2)
    return cost


def cost_function_SIM(tunables, *args):
    eskf_parameters = tunables[:4]
    eskf_parameters = np.append(eskf_parameters, args[0])
    eskf_parameters = np.abs(eskf_parameters)
    p_std = np.abs(np.repeat(tunables[4:6], [2, 1]))

    P_pred_init_list, x_init, loaded_data, N, offset, use_GNSSaccuracy = args[1:]
    try:
        result = run_eskf(eskf_parameters, x_init, P_pred_init_list, loaded_data,
                          p_std, N, offset=offset,
                          use_GNSSaccuracy=use_GNSSaccuracy)
        # delta_x = result[3]

        NIS = result[4]
        NEES_list = result[6:11]

        def log_error_func(x):
            return np.mean(np.log(x)**2)
        # cost = np.mean(np.log(NIS[:, 1])**2)
        delta = result[3]
        cost_delta = np.mean(np.sum(delta**2, axis=1))
        nees_all = result[5]
        cost_nees = sum([log_error_func(i) for i in NEES_list])
        cost_nis = np.mean(np.log(NIS[:, 1])**2)
        cost = 20*cost_delta + cost_nees + cost_nis

        with open('optimization.txt', 'a') as file:
            text = (f'eskf_parameters: {tunables[:4]}\n'
                    f'gps_parameters: {tunables[4:]}\n'
                    f'cost {cost_delta}, {cost_nees}, {cost_nis}\n'
                    f'{cost}\n\n')
            file.write(text)
            print(text)
            time.sleep(0.2)
        return cost

    except:
        return np.inf


def optimize(cost_function, eskf_parameters, p_std,
             x_pred_init, P_pred_init_list, loaded_data, N, offset,
             use_GNSSaccuracy=True):
    with open('optimization.txt', 'a') as file:
        file.write(f'\n\nNew session\n')

    tunables_init = eskf_parameters[:-2] + list(p_std[1:])
    extra_args = ([eskf_parameters[-2:]] + [P_pred_init_list]
                  + [x_pred_init] + [loaded_data, N, offset, use_GNSSaccuracy])
    minimize(cost_function, tunables_init, tuple(
        extra_args), options={'disp': True})
