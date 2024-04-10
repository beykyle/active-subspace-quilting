import numpy as np
import pickle
from tqdm import tqdm
from pathlib import Path
from scipy.spatial import kdtree

import rose
from rose.training import latin_hypercube_sample
from rose.koning_delaroche import EnergizedKoningDelaroche
from rose import ScatteringAmplitudeEmulator

l_max=10
angles = np.linspace(0, np.pi, 10)
s_mesh = np.linspace(1e-2, 6 * np.pi, 1200)
domain = np.array([s_mesh[0], s_mesh[-1]])
s_0 = 5.5 * np.pi
bounds = np.load("./kd_ff_bounds.npy")
train = np.load("./kd_ff_train.npy")
frozen_params = bounds[:,1] == bounds[:,0]
unfrozen_mask = np.logical_not(frozen_params)

# use log(E) space
scaleE = 1.0
def forward_pspace_transform(sample):
    return np.hstack([ np.log(sample[0]/scaleE), sample[1:]])

def backward_pspace_transform(sample):
    return np.hstack([np.exp(sample[0])*scaleE, sample[1:]])


interactions = rose.koning_delaroche.EnergizedKoningDelaroche(
    training_info=bounds,
    n_basis=15,
    l_max=l_max,
    n_train=10,
)
sae = rose.ScatteringAmplitudeEmulator.HIFI_solver(
    base_solver=rose.SchroedingerEquation.make_base_solver(
        s_0=s_0,
        domain=domain,
    ),
    interaction_space=interactions,
    angles=angles,
    s_mesh=s_mesh,
)

asq = rose.ActiveSubspaceQuilt(
    interactions,
    sae,
    s_mesh,
    s_0,
    bounds,
    train,
    None,
    None,
    frozen_params,
    70,
    0.05,
    threads=8,
)
np.save("hf_solns_uq_log.npy", asq.hf_solns)
import pickle
with open("./asq_emulators.pkl", "wb") as f:
    pickle.dump(asq.emulators, f)

