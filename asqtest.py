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
angles = np.linspace(0, np.pi, 200)
s_mesh = np.linspace(1e-2, 8 * np.pi, 1000)
domain = np.array([s_mesh[0], s_mesh[-1]])
s_0 = 7 * np.pi
params =  np.load("kd_ff_params.npy")
lower_bound = np.min(params, axis=(0, 1))
upper_bound = np.max(params, axis=(0, 1))
bounds = np.vstack([lower_bound, upper_bound]).T
frozen_params = bounds[:,1] == bounds[:,0]
unfrozen_mask = np.logical_not(frozen_params)
param_labels  =np.asarray(
    [
        r"$E$",
        r"$\mu$",
        r"$v_v$",
        r"$r_v$",
        r"$a_v$",
        r"$v_w$",
        r"$r_{w}$",
        r"$a_{w}$",
        r"$v_d$",
        r"$r_{d}$",
        r"$a_d$",
        r"$v_{so}$",
        r"$r_{so}$",
        r"$a_{so}$",
        r"$w_{so}$",
        r"$r_{wso}$",
        r"$a_{wso}$",
    ]

)

n_train = 3000
n_test = 500
train = latin_hypercube_sample(n_train, bounds, seed=133)
test = latin_hypercube_sample(n_test, bounds, seed=111)

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
    50,
    0.1,
    threads=8,
)

asq.save("asq.pkl")
