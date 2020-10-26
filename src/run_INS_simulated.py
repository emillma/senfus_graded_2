# %% imports
from scipy.optimize import minimize
import scipy
import scipy.io
import scipy.stats

import matplotlib.pyplot as plt
import numpy as np
import os
import time
from zlib import adler32
from plotter import plot_path, plot_estimate, state_error_plots
from eskf_runner import run_eskf

from eskf import (
    POS_IDX,
    VEL_IDX,
)

from quaternion import quaternion_to_euler
from cat_slice import CatSlice

# %% plot config check and style setup

from plott_setup import setup_plot
setup_plot()

# %% load data and plot
folder = os.path.dirname(__file__)
# filename_to_load = f"{folder}/../data/task_simulation.mat"
filename_to_load = f"{folder}/../data/task_simulation.mat"
cache_folder = os.path.join(folder, '..', 'cache')
loaded_data = scipy.io.loadmat(filename_to_load)

timeIMU = loaded_data["timeIMU"].ravel()
if 'xtrue' in loaded_data:
    x_true = loaded_data["xtrue"].T
else:
    x_true = None
z_GNSS = loaded_data["zGNSS"].T
dt = np.mean(np.diff(timeIMU))


# %% Measurement noise
# IMU noise values for STIM300, based on datasheet and simulation sample rate
# Continous noise
# TODO: What to remove here?
cont_gyro_noise_std = 4.36e-5  # (rad/s)/sqrt(Hz)
cont_acc_noise_std = 1.167e-3  # (m/s**2)/sqrt(Hz)

# Discrete sample noise at simulation rate used
# Hvorfor gange med en halv? (eq. 10.70)
acc_std = 0.5 * cont_acc_noise_std * np.sqrt(1 / dt)
rate_std = 0.5 * cont_gyro_noise_std * np.sqrt(1 / dt)

# Bias values
acc_bias_driving_noise_std = 4e-3
cont_acc_bias_driving_noise_std = 6 * \
    acc_bias_driving_noise_std / np.sqrt(1 / dt)

rate_bias_driving_noise_std = 5e-5
cont_rate_bias_driving_noise_std = (
    (1/3) * rate_bias_driving_noise_std / np.sqrt(1 / dt)
)


# Position and velocity measurement
p_std = np.array([0.3, 0.3, 0.5])  # Measurement noise

p_acc = 1e-16
p_gyro = 1e-16
# [-2.71073648e-02  1.97296299e-04 -7.88136014e-04  5.62588030e-05
#  -7.26577980e-04  1.43524292e-01]
eskf_parameters = [acc_std,
                   rate_std,
                   cont_acc_bias_driving_noise_std,
                   cont_rate_bias_driving_noise_std,
                   p_acc,
                   p_gyro]
# %% Initialise
x_pred_init = np.zeros(16)
x_pred_init[POS_IDX] = np.array([0, 0, -5])  # starting 5 metres above ground
x_pred_init[VEL_IDX] = np.array([20, 0, 0])  # starting at 20 m/s due north
# no initial rotation: nose to North, right to East, and belly down
x_pred_init[6] = 1

# These have to be set reasonably to get good results

# [241.94198986 319.04325528   2.06011741   0.77419239   0.53211694] best result so far with simplex
P_pred_init_pos = 10
P_pred_init_vel = 10
P_pred_init_err_att = 1
P_pred_init_err_acc_bias = 0.1
P_pred_init_err_gyro_bias = 0.1
P_pred_init_list = [P_pred_init_pos,
                    P_pred_init_vel,
                    P_pred_init_err_att,
                    P_pred_init_err_acc_bias,
                    P_pred_init_err_gyro_bias]


init_parameters = [x_pred_init, P_pred_init_list]

# %% Run estimation

N: int = 5000
doGNSS: bool = True
# TODO: Set this to False if you want to check that the predictions make sense over reasonable time lenghts


use_cache = True
parameters = eskf_parameters + init_parameters
parameter_hash = str(adler32(str(parameters + [N, doGNSS]).encode()))
# %% plotting


def cost_function(x, *args):
    P_pred_init_list = x

    print(x)
    eskf_parameters, x_init, loaded_data, p_std, N = args
    result = run_eskf(eskf_parameters, x_init, P_pred_init_list, loaded_data,
                      p_std, N)
    delta_x = result[3]
    rmse = np.sqrt(np.mean(np.sum(delta_x[:N, :3]**2, axis=1)))
    print(f'RMSE: {rmse}\n')
    time.sleep(0.2)
    return rmse


initial_guess = P_pred_init_list
extra_args = [eskf_parameters] + [x_pred_init] + [loaded_data, p_std, N]
# minimize(cost_function, initial_guess, tuple(extra_args))
(x_pred,
    x_est,
    P_est,
    delta_x,
    NEES_all,
    NEES_pos,
    NEES_vel,
    NEES_att,
    NEES_accbias,
    NEES_gyrobias,
    GNSSk) = run_eskf(eskf_parameters, x_pred_init, P_pred_init_list, loaded_data,
                      p_std, N, doGNSS=doGNSS)


t = np.linspace(0, dt * (N - 1), N)
plot_path(N, GNSSk, x_est, z_GNSS, x_true)
plot_estimate(t, N, x_est)
state_error_plots(t, N, x_est, x_true, delta_x)
plt.show()
