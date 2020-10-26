# %% imports
import scipy
import scipy.io
import scipy.stats

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import tqdm
import os
import pickle
from zlib import adler32
from plotter import plot_traj, plot_estimate
from run_eskf import run_eskf
# try:  # see if tqdm is available, otherwise define it as a dummy
#     try:  # Ipython seem to require different tqdm.. try..except seem to be the easiest way to check
#         __IPYTHON__
#         import tqdm
#     except:
#         import tqdm
# except Exception as e:
#     print(e)S
#     print(
#         "install tqdm (conda install tqdm, or pip install tqdm) to get nice progress bars. "
#     )

#     def tqdm(iterable, *args, **kwargs):
#         return iterable

from eskf import (
    ESKF,
    POS_IDX,
    VEL_IDX,
    ATT_IDX,
    ACC_BIAS_IDX,
    GYRO_BIAS_IDX,
    ERR_ATT_IDX,
    ERR_ACC_BIAS_IDX,
    ERR_GYRO_BIAS_IDX,
)

from quaternion import quaternion_to_euler
from cat_slice import CatSlice

# %% plot config check and style setup

from plott_setup import setup_plot
setup_plot()

# %% load data and plot
folder = os.path.dirname(__file__)
filename_to_load = f"{folder}/../data/task_simulation.mat"
cache_folder = os.path.join(folder, '..', 'cache')
loaded_data = scipy.io.loadmat(filename_to_load)

# S_a = loaded_data["S_a"]
# S_g = loaded_data["S_g"]
# lever_arm = loaded_data["leverarm"].ravel()
# timeGNSS = loaded_data["timeGNSS"].ravel()
timeIMU = loaded_data["timeIMU"].ravel()
# x_true = loaded_data["xtrue"].T
# z_acceleration = loaded_data["zAcc"].T
# z_GNSS = loaded_data["zGNSS"].T
# z_gyroscope = loaded_data["zGyro"].T
# Ts_IMU = [0, *np.diff(timeIMU)]
dt = np.mean(np.diff(timeIMU))
# steps = len(z_acceleration)
# gnss_steps = len(z_GNSS)

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
R_GNSS = np.diag(p_std ** 2)

p_acc = 1e-16
p_gyro = 1e-16
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
P_pred_init = np.zeros((15, 15))
P_pred_init[POS_IDX ** 2] = 10**2 * np.eye(3)
P_pred_init[VEL_IDX ** 2] = 10**2 * np.eye(3)
P_pred_init[ERR_ATT_IDX ** 2] = np.eye(3)
P_pred_init[ERR_ACC_BIAS_IDX ** 2] = 0.01 * np.eye(3)
P_pred_init[ERR_GYRO_BIAS_IDX ** 2] = 0.01 * np.eye(3)

init_parameters = [x_pred_init, P_pred_init]

# %% Run estimation
# run this file with 'python -O run_INS_simulated.py' to turn of assertions and get about 8/5 speed increase for longer runs

N: int = 5000
doGNSS: bool = True
# TODO: Set this to False if you want to check that the predictions make sense over reasonable time lenghts

use_cache = True
parameters = eskf_parameters + init_parameters
parameter_hash = str(adler32(str(parameters + [N, doGNSS]).encode()))
# if parameter_hash not in os.listdir(cache_folder) or not use_cache:


#     if use_cache:
#         with open(os.path.join(cache_folder, parameter_hash), 'wb') as file:
#             pickle.dump(result, file)
# else:
#     with open(os.path.join(cache_folder, parameter_hash), 'rb') as file:
#         (x_pred,
#          x_est,
#          P_est,
#          NEES_all,
#          NEES_pos,
#          NEES_vel,
#          NEES_att,
#          NEES_accbias,
#          NEES_gyrobias,
#          k) = pickle.load(file)
# %% plotting
(x_pred,
    x_est,
    P_est,
    NEES_all,
    NEES_pos,
    NEES_vel,
    NEES_att,
    NEES_accbias,
    NEES_gyrobias,
    GNSSk) = run_eskf(eskf_parameters, init_parameters, loaded_data,
                      R_GNSS, N)


dosavefigures = False
doplothandout = False

t = np.linspace(0, dt * (N - 1), N)
plot_traj(N, GNSSk, x_est, x_true, z_GNSS)
plot_estimate(t, N, x_est)
plt.show()
