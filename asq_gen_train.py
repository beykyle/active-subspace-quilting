import numpy as np
import pickle
from tqdm import tqdm
from pathlib import Path
from scipy.spatial import kdtree

import rose
from rose.koning_delaroche import KD_simple, KD_simple_so, KDGlobal, Projectile
from rose.training import latin_hypercube_sample

cgmf_ebin = np.linspace(0.01, 10, 200)
ebounds = np.array([cgmf_ebin[0], cgmf_ebin[-1]])
E_GRID_SIZE = cgmf_ebin.size
E_GRID = cgmf_ebin
isotopes = [
    (56,134),
    (56,135),
    (56,136),
    (56,137),
    (48,110),
    (48,111),
    (48,112),
    (48,113),
    (48,114),
    (48,116),
    (48,106),
    (48,108),
    (55,133),
    (53,127),
    (49,115),
    (57,139),
    (42,92),
    (42,94),
    (42,96),
    (42,98),
    (42,100),
    (42,95),
    (41,93),
    (45,103),
    (50,120),
    (50,118),
    (50,122),
    (50,124),
    (50,116),
    (50,114),
    (50,115),
    (50,117),
    (50,112),
    (50,119),
    (43,99),
    (52,122),
    (52,124),
    (52,125),
    (52,126),
    (52,128),
    (52,130),
    (40,90),
]
NUM_ISOTOPES = len(isotopes)
omp = rose.koning_delaroche.KDGlobal(Projectile.neutron)
(mu, Ecom, k, eta, R_C), parameters = omp.get_params(1, 1, 0.1)
N_PARAMS = len(parameters) + 2
N_PARAMS
params = np.zeros((NUM_ISOTOPES, E_GRID_SIZE, N_PARAMS), dtype=np.double)
for i, (Z,A) in tqdm(enumerate(isotopes), total=NUM_ISOTOPES):
    for j, e in enumerate(E_GRID):
        (mu, Ecom, _, _, _), parameters = omp.get_params(A, Z, E_com=e)
        params[i,j,:] = np.array([Ecom, mu, *parameters])
np.save("./kd_ff_params.npy", params)

lower_bound = np.min(params, axis=(0, 1))
upper_bound = np.max(params, axis=(0, 1))
bounds = np.vstack([lower_bound, upper_bound]).T
frozen_params = bounds[:,1] == bounds[:,0]
unfrozen_mask = np.logical_not(frozen_params)
np.save("./kd_ff_bounds.npy", bounds)

n_train = 3000
n_train_log = 500
n_test = 1000

train = latin_hypercube_sample(n_train, bounds, seed=137)

log_energy_bounds = bounds.copy()
log_energy_bounds[0,:] = np.log(bounds[0,:])
log_train = latin_hypercube_sample(n_train_log, log_energy_bounds, seed=139)
log_train[:,0] = np.exp(log_train[:,0])

train = np.concatenate([ train, log_train])
np.save("kd_ff_train.npy", train)
