import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
from pathlib import Path
import time
from jax import random as jr
import klax
from dynax import ISPHS, ConvexLyapunov, ODESolver, PODLatentSpace
from jax.nn.initializers import variance_scaling

# CONFIGURATION
SEED = 42
LATENT_DIM = 16
U_MODES = 40
N_TRAIN = 4
N_TRAJ = 25
LR = 1e-3
STEPS_P1 = 10000
STEPS_P2 = 4000
BATCH_P1 = 8
BATCH_P2 = 4
DATA_DIR = Path("ded_data")
MODEL_PATH = Path("DED_POD_final.eqx")
print(f"POD + sPHNN_DED")

# HELPERS
def to_step(x):
    shape = x.shape
    x2 = jnp.stack([x[:, :-1, ...], x[:, 1:, ...]], axis=2)
    return x2.reshape(-1, 2, *shape[2:])

def trajectory_loss(model, data, batch_axis):
    ts, ys, us = data
    ta, ya, ua = batch_axis
    ys_pred = jax.vmap(model, in_axes=(ta, ya, ua))(ts, ys[:, 0], us)
    return jnp.mean((ys_pred - ys) ** 2)

def build_model(state_size, input_size, key):
    ficnn_key, h_key, j_key, r_key, g_key = jr.split(key, 5)
    ficnn = klax.nn.FICNN(
        in_size=state_size, out_size="scalar", width_sizes=[16, 16],
        weight_init=variance_scaling(1, "fan_avg", "truncated_normal"), key=ficnn_key)
    deriv_model = ISPHS(
        ConvexLyapunov(ficnn, state_size=state_size, key=h_key),
        klax.nn.ConstantSkewSymmetricMatrix(state_size, init=variance_scaling(1, "fan_avg", "truncated_normal"), key=j_key),
        klax.nn.ConstantSPDMatrix(state_size, epsilon=0.0, init=variance_scaling(1, "fan_avg", "truncated_normal"), key=r_key),
        klax.nn.ConstantMatrix((state_size, input_size), init=variance_scaling(1, "fan_avg", "truncated_normal"), key=g_key))
    return ODESolver(deriv_model)

# LOAD DATA
t0 = time.time()
data = np.load(DATA_DIR / "data.npz")
ts = data["ts"]
ys = data["temp"]
us = data["source"]
num_trajectories = ys.shape[0]
N_TRAJ = min(25, num_trajectories)
ys = ys[:N_TRAJ]; us = us[:N_TRAJ]
 
# Split
split_key = jr.key(0); _, subkey = jr.split(split_key)
indices = jr.permutation(subkey, jnp.arange(N_TRAJ))
train_idx = np.array(indices[:N_TRAIN]); test_idx = np.array(indices[N_TRAIN:])
ys_train = ys[train_idx]; ys_test = ys[test_idx]
us_train = us[train_idx]; us_test = us[test_idx]
print(f"Train idx: {train_idx} | Test idx: {test_idx}")
 
# POD
y_pod = PODLatentSpace(ys_train, num_modes=LATENT_DIM, shift=293)
u_pod = PODLatentSpace(us_train, num_modes=U_MODES)
 
ys_latent_train = y_pod.to_latent(ys_train)
ys_latent_test = y_pod.to_latent(ys_test)
us_latent_train = u_pod.to_latent(us_train)
us_latent_test = u_pod.to_latent(us_test)
 
compress_rmse = float(jnp.sqrt(jnp.mean((ys - np.array(y_pod.from_latent(y_pod.to_latent(ys)))) ** 2)))
print(f"POD Compression RMSE: {compress_rmse:.3e}")

# Single-step data
ts_step_train = to_step(jnp.tile(ts, (ys_train.shape[0], 1)))
ys_latent_step_train = to_step(ys_latent_train)
us_latent_step_train = to_step(us_latent_train)

key = jr.PRNGKey(SEED)
key, model_key = jr.split(key)
model = build_model(LATENT_DIM, U_MODES, model_key)

if MODEL_PATH.exists():
    print(f"  Loading: {MODEL_PATH}")
    model = eqx.tree_deserialise_leaves(MODEL_PATH, model)
else:
    t_run = time.time()

    # Phase 1: Single-Step
    key, train_key = jr.split(key)
    model, _ = klax.fit(
        model,
        (ts_step_train, ys_latent_step_train, us_latent_step_train),
        batch_size=128,
        batch_axis=(0, 0, 0),
        steps=10000, 
        loss_fn=trajectory_loss,
        history=klax.HistoryCallback(log_every=100),
        key=train_key)

    # Phase 2: Full Trajectory
    key, finetune_key = jr.split(key)
    model, _ = klax.fit(
        model,
        (ts, ys_latent_train, us_latent_train),
        batch_size= 4,
        batch_axis=(None, 0, 0),
        steps=4000, 
        loss_fn=trajectory_loss,
        history=klax.HistoryCallback(log_every=100),
        key=finetune_key)

    eqx.tree_serialise_leaves(MODEL_PATH, model)
    print(f"  Trained in {(time.time()-t_run)/60:.1f} min -> {MODEL_PATH}")

# Evaluate
final = klax.finalize(model)

ys_lat_train_pred = jax.vmap(final, in_axes=(None, 0, 0))(ts, ys_latent_train[:, 0], us_latent_train)
ys_train_pred = np.array(y_pod.from_latent(ys_lat_train_pred))
ode_rmse_train = float(np.sqrt(np.mean((ys_train - ys_train_pred)**2)))

ys_lat_test_pred = jax.vmap(final, in_axes=(None, 0, 0))(ts, ys_latent_test[:, 0], us_latent_test)
ys_test_pred = np.array(y_pod.from_latent(ys_lat_test_pred))
ode_rmse_test = float(np.sqrt(np.mean((ys_test - ys_test_pred)**2)))

result = {'n_train': n_train, 'compress_rmse': compress_rmse,
          'ode_rmse_train': ode_rmse_train, 'ode_rmse_test': ode_rmse_test}
all_results.append(result)
print(f"  ODE Train: {ode_rmse_train:.2f} K | ODE Test: {ode_rmse_test:.2f} K")
print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
