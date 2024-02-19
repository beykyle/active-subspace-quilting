import rose
from rose.training import latin_hypercube_sample
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
import jitr
from numba import njit

from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

MESH_SIZE = 2000
LMAX = 15
NSAMPLES = 80

SAMPLES_PER_RANK = NSAMPLES // size

solutions_spin_up = np.empty(
    (SAMPLES_PER_RANK, LMAX, MESH_SIZE), dtype=np.complex128
)
solutions_spin_down = np.empty(
    (SAMPLES_PER_RANK, LMAX - 1, MESH_SIZE), dtype=np.complex128
)


params = np.load("./global_kd_params.npy")
lower_bound = np.min(params, axis=(0, 1))
upper_bound = np.max(params, axis=(0, 1))
bounds = np.vstack([lower_bound, upper_bound]).T
train = latin_hypercube_sample(NSAMPLES, bounds, seed=13)


interactions = rose.koning_delaroche.EnergizedKoningDelaroche(
    bounds,
    n_basis=10,
    l_max=LMAX,
    n_train=10,
)
solver = rose.ScatteringAmplitudeEmulator.HIFI_solver(
    interactions,
    angles=np.linspace(0, np.pi, 200),
    s_mesh=np.linspace(1e-6, 6.1 * np.pi, MESH_SIZE),
)


idx = 0
for i in range(rank * SAMPLES_PER_RANK, (rank + 1) * SAMPLES_PER_RANK):
    sample = train[i]
    # solns = solver.exact_wave_functions(sample)
    for l in range(LMAX):
        solutions_spin_up[idx, l, :] = np.ones(MESH_SIZE) * i
        if l > 0:
            solutions_spin_down[idx, l - 1, :] = np.ones(MESH_SIZE) * i
    idx = idx + 1


ru = comm.gather(solutions_spin_up, root=0)
rd = comm.gather(solutions_spin_down, root=0)

if rank == 0:
    solutions_spin_up = np.concatenate(ru, axis=0)
    solutions_spin_down = np.concatenate(rd, axis=0)

np.save("training_solutions_su_kd_global.npy", solutions_spin_up)
np.save("training_solutions_sd_kd_global.npy", solutions_spin_down)
