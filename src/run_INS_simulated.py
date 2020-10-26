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
from plotter import plot_traj
# try:  # see if tqdm is available, otherwise define it as a dummy
#     try:  # Ipython seem to require different tqdm.. try..except seem to be the easiest way to check
#         __IPYTHON__
#         import tqdm
#     except:
#         import tqdm
# except Exception as e:
#     print(e)
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
loaded_data = scipy.io.loadmat(filename_to_load)

S_a = loaded_data["S_a"]
S_g = loaded_data["S_g"]
lever_arm = loaded_data["leverarm"].ravel()
timeGNSS = loaded_data["timeGNSS"].ravel()
timeIMU = loaded_data["timeIMU"].ravel()
x_true = loaded_data["xtrue"].T
z_acceleration = loaded_data["zAcc"].T
z_GNSS = loaded_data["zGNSS"].T
z_gyroscope = loaded_data["zGyro"].T

Ts_IMU = [0, *np.diff(timeIMU)]

dt = np.mean(np.diff(timeIMU))
steps = len(z_acceleration)
gnss_steps = len(z_GNSS)

# %% Measurement noise
# IMU noise values for STIM300, based on datasheet and simulation sample rate
# Continous noise
# TODO: What to remove here?
cont_gyro_noise_std = 4.36e-5  # (rad/s)/sqrt(Hz)
cont_acc_noise_std = 1.167e-3  # (m/s**2)/sqrt(Hz)

# Discrete sample noise at simulation rate used
# Hvorfor gange med en halv? (eq. 10.70)
rate_std = 0.5 * cont_gyro_noise_std * np.sqrt(1 / dt)
acc_std = 0.5 * cont_acc_noise_std * np.sqrt(1 / dt)

# Bias values
rate_bias_driving_noise_std = 5e-5
cont_rate_bias_driving_noise_std = (
    (1/3) * rate_bias_driving_noise_std / np.sqrt(1 / dt)
)

acc_bias_driving_noise_std = 4e-3
cont_acc_bias_driving_noise_std = 6 * \
    acc_bias_driving_noise_std / np.sqrt(1 / dt)

# Position and velocity measurement
p_std = np.array([0.3, 0.3, 0.5])  # Measurement noise
R_GNSS = np.diag(p_std ** 2)

p_acc = 1e-16
p_gyro = 1e-16

# %% Estimator
eskf = ESKF(
    acc_std,
    rate_std,
    cont_acc_bias_driving_noise_std,
    cont_rate_bias_driving_noise_std,
    p_acc,
    p_gyro,
    S_a=S_a,  # set the accelerometer correction matrix
    S_g=S_g,  # set the gyro correction matrix,
    debug=True  # TODO: False to avoid expensive debug checks, can also be suppressed by calling 'python -O run_INS_simulated.py'
)

# %% Allocate
x_est = np.zeros((steps, 16))
P_est = np.zeros((steps, 15, 15))

x_pred = np.zeros((steps, 16))
P_pred = np.zeros((steps, 15, 15))

delta_x = np.zeros((steps, 15))

NIS = np.zeros(gnss_steps)

NEES_all = np.zeros(steps)
NEES_pos = np.zeros(steps)
NEES_vel = np.zeros(steps)
NEES_att = np.zeros(steps)
NEES_accbias = np.zeros(steps)
NEES_gyrobias = np.zeros(steps)

# %% Initialise
x_pred[0, POS_IDX] = np.array([0, 0, -5])  # starting 5 metres above ground
x_pred[0, VEL_IDX] = np.array([20, 0, 0])  # starting at 20 m/s due north
# no initial rotation: nose to North, right to East, and belly down
x_pred[0, 6] = 1

# These have to be set reasonably to get good results
P_pred[0][POS_IDX ** 2] = 10**2 * np.eye(3)
P_pred[0][VEL_IDX ** 2] = 10**2 * np.eye(3)
P_pred[0][ERR_ATT_IDX ** 2] = np.eye(3)
P_pred[0][ERR_ACC_BIAS_IDX ** 2] = 0.01 * np.eye(3)
P_pred[0][ERR_GYRO_BIAS_IDX ** 2] = 0.01 * np.eye(3)

# %% Run estimation
# run this file with 'python -O run_INS_simulated.py' to turn of assertions and get about 8/5 speed increase for longer runs

N: int = 5000
# TODO: Set this to False if you want to check that the predictions make sense over reasonable time lenghts
doGNSS: bool = False

GNSSk: int = 0  # keep track of current step in GNSS measurements
for k in tqdm.trange(N):
    if doGNSS and timeIMU[k] >= timeGNSS[GNSSk]:
        NIS[GNSSk] = eskf.NIS_GNSS_position(
            x_pred[k], P_pred[k], z_GNSS[GNSSk], R_GNSS, lever_arm)

        x_est[k], P_est[k] = eskf.update_GNSS_position(
            x_pred[k], P_pred[k], z_GNSS[GNSSk], R_GNSS, lever_arm)
        assert np.all(np.isfinite(P_est[k])), f"Not finite P_pred at index {k}"

        GNSSk += 1
    else:
        # no updates, so let us take estimate = prediction
        x_est[k] = x_pred[k]
        P_est[k] = P_pred[k]

    delta_x[k] = eskf.delta_x(x_est[k], x_true[k])
    (
        NEES_all[k],
        NEES_pos[k],
        NEES_vel[k],
        NEES_att[k],
        NEES_accbias[k],
        NEES_gyrobias[k],
    ) = eskf.NEESes(x_est[k], P_est[k], x_true[k])

    if k < N - 1:
        x_pred[k + 1], P_pred[k + 1] = eskf.predict(
            x_est[k], P_est[k], z_acceleration[k+1], z_gyroscope[k+1], Ts_IMU[k+1])

    if eskf.debug:
        assert np.all(np.isfinite(P_pred[k])
                      ), f"Not finite P_pred at index {k + 1}"


# %% plotting
dosavefigures = False
doplothandout = False

t = np.linspace(0, dt * (N - 1), N)
plot_traj(N, GNSSk, x_est, x_true, z_GNSS)
plt.show()
