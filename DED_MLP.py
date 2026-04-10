import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
import pyvista as pv
from pathlib import Path
import time
from jax import random as jr
import klax
from dynax import ISPHS, ConvexLyapunov, ODESolver
from jax.nn.initializers import variance_scaling
from sklearn.cluster import KMeans

# DATA LOADING
def load_ded_data(data_dir, downsample=1, trajectories=None):
    raw_data     = np.load(data_dir / "data.npz")
    nodes        = np.array(raw_data["nodes"])
    temperatures = np.array(raw_data["temp"])
    sources      = np.array(raw_data["source"])
    ts           = np.array(raw_data["ts"])
    n_trajectories, n_timesteps, n_nodes_orig = temperatures.shape
    if downsample > 1:
        nodes = nodes[::downsample]; temperatures = temperatures[:,:,::downsample]
        sources = sources[:,:,::downsample]
    if trajectories is None: trajectories = list(range(n_trajectories))
    temp_sel = temperatures[trajectories]; source_sel = sources[trajectories]
    n_nodes = nodes.shape[0]
    temp_shift = 293.0; temp_scale = float(temp_sel.std())
    source_max = float(source_sel.max())
    if source_max < 1e-8: source_max = 1.0
    stats = {'temp_shift': jnp.array(temp_shift), 'temp_scale': jnp.array(temp_scale),
             'source_max': jnp.array(source_max), 'n_nodes': n_nodes}
    temp_norm = (temp_sel - temp_shift) / temp_scale
    source_norm = source_sel / source_max
    all_features = jnp.array(np.concatenate([temp_norm[:,:,:,None], source_norm[:,:,:,None]], axis=-1).astype(np.float32))
    metadata = {'n_trajectories': len(trajectories), 'n_timesteps': n_timesteps, 'n_nodes': n_nodes}
    print(f"   {len(trajectories)} Traj | {n_nodes} Nodes")
    return all_features, jnp.array(ts), stats, metadata

# DATA PREPARATION

def prepare_phase1_data(train_features, ts):
    n_traj, T, n_nodes, n_feat = train_features.shape
    all_ts, all_init, all_seq = [], [], []
    for i in range(n_traj):
        for t in range(T - 1):
            all_ts.append(jnp.array([ts[t], ts[t+1]]))
            all_init.append(train_features[i, t])
            all_seq.append(train_features[i, t:t+2])
    return jnp.stack(all_ts), jnp.stack(all_init), jnp.stack(all_seq)

def prepare_phase2_data(train_features, ts):
    n_traj = train_features.shape[0]
    return jnp.stack([ts] * n_traj), train_features[:, 0], train_features

# MLP AUTOENCODER

class MLPAutoEncoder(eqx.Module):
    encoder_mlp: eqx.nn.MLP; decoder_mlp: eqx.nn.MLP
    n_nodes: int = eqx.field(static=True); latent_dim: int = eqx.field(static=True)

    def __init__(self, n_nodes, latent_dim, key):
        keys = jr.split(key, 2)
        self.n_nodes = n_nodes; self.latent_dim = latent_dim
        self.encoder_mlp = eqx.nn.MLP(n_nodes, latent_dim, 512, 3, activation=jnp.tanh, key=keys[0])
        self.decoder_mlp = eqx.nn.MLP(latent_dim, n_nodes, 512, 3, activation=jnp.tanh,
                                       final_activation=lambda x: x, key=keys[1])

    def _encode_raw(self, temp): return self.encoder_mlp(temp)
    def encode(self, nodes):
        temp = nodes[:, 0]
        return self._encode_raw(temp) - self._encode_raw(jnp.zeros_like(temp))
    def _decode_raw(self, z): return self.decoder_mlp(z).reshape(-1, 1)
    def decode(self, z): return self._decode_raw(z) - self._decode_raw(jnp.zeros_like(z))
    def __call__(self, nodes): return self.decode(self.encode(nodes))

# LASER ENCODER + sPHNN + FULL MODEL

class LaserInputEncoder(eqx.Module):
    mlp: eqx.nn.MLP; output_dim: int = eqx.field(static=True)
    def __init__(self, key, n_nodes, output_dim=8):
        self.output_dim = output_dim
        self.mlp = eqx.nn.MLP(n_nodes, output_dim, 64, 2, activation=jnp.tanh, key=key)
    def __call__(self, source):
        return self.mlp(source) - self.mlp(jnp.zeros_like(source))

class sPHNNDynamics(eqx.Module):
    deriv_model: ISPHS; solver: ODESolver; laser_encoder: LaserInputEncoder
    def __init__(self, key, state_size, laser_latent_size=8, n_nodes=1836):
        ficnn_key, h_key, j_key, r_key, g_key, laser_key = jr.split(key, 6)
        ficnn = klax.nn.FICNN(state_size, "scalar", [16, 16],
                               weight_init=variance_scaling(1, "fan_avg", "truncated_normal"), key=ficnn_key)
        self.deriv_model = ISPHS(
            ConvexLyapunov(ficnn, state_size=state_size, key=h_key),
            klax.nn.ConstantSkewSymmetricMatrix(state_size, init=variance_scaling(1, "fan_avg", "truncated_normal"), key=j_key),
            klax.nn.ConstantSPDMatrix(state_size, epsilon=0.0, init=variance_scaling(1, "fan_avg", "truncated_normal"), key=r_key),
            klax.nn.ConstantMatrix((state_size, laser_latent_size), init=variance_scaling(1, "fan_avg", "truncated_normal"), key=g_key))
        self.solver = ODESolver(self.deriv_model)
        self.laser_encoder = LaserInputEncoder(laser_key, n_nodes=n_nodes, output_dim=laser_latent_size)
    def __call__(self, ts, z0, source_sequence):
        return self.solver(ts, z0, jax.vmap(self.laser_encoder)(source_sequence))

class MLPsPHNNFullModel(eqx.Module):
    ae: MLPAutoEncoder; dynamics: sPHNNDynamics
    def __init__(self, key, ae_architecture):
        self.ae = ae_architecture
        self.dynamics = sPHNNDynamics(key, state_size=ae_architecture.latent_dim,
                                     laser_latent_size=8, n_nodes=ae_architecture.n_nodes)


# LOSS
RECON_WEIGHT = 0.5
def trajectory_loss_combined(model, data, batch_axis):
    ts_batch, nodes_t, nodes_seq = data
    def predict_single(ts_i, init_nodes, seq_nodes):
        z0 = model.ae.encode(init_nodes)
        source_seq = seq_nodes[:, :, 1]
        zs = model.dynamics(ts_i, z0, source_seq)
        preds = jax.vmap(model.ae.decode)(zs)
        target_temp = seq_nodes[:, :, 0:1]
        dyn_loss = jnp.mean((preds - target_temp) ** 2)
        def recon_single(nodes):
            recon = model.ae(nodes)
            return jnp.mean((recon - nodes[:, 0:1])**2)
        recon_loss = jnp.mean(jax.vmap(recon_single)(seq_nodes))
        return dyn_loss + RECON_WEIGHT * recon_loss
    losses = jax.vmap(predict_single)(ts_batch, nodes_t, nodes_seq)
    return jnp.mean(losses)


# EVALUATION

def compute_mlp_metrics(model, features, ts, scale, shift):
    ae_rmses, ode_rmses = [], []
    for ti in range(features.shape[0]):
        traj = features[ti]
        zs = jax.vmap(model.ae.encode)(traj)
        recon = jax.vmap(lambda z: model.ae.decode(z).squeeze())(zs)
        ae_rmses.append(float(jnp.sqrt(jnp.mean((recon - traj[:, :, 0])**2))) * scale)
        z0 = model.ae.encode(traj[0])
        zs_pred = model.dynamics(ts, z0, traj[:, :, 1])
        pred = jax.vmap(lambda z: model.ae.decode(z).squeeze())(zs_pred)
        pred_k = np.array(pred) * scale + shift
        true_k = np.array(traj[:, :, 0]) * scale + shift
        ode_rmses.append(float(np.sqrt(np.mean((pred_k - true_k)**2))))
    return float(np.mean(ae_rmses)), float(np.mean(ode_rmses))


# MAIN

def main():
    STEPS_PHASE1 = 10000
    STEPS_PHASE2 = 4000
    BATCH_P1     = 128
    BATCH_P2     = 4
    LR_P1        = 9e-5
    LR_P2        = 1e-4
    N_TRAJ       = 25
    N_TRAIN      = 4
    LATENT_DIM   = 16
    SEED = 42
    DATA_DIR = Path("ded_data")
    MODEL_PATH = Path("DED_MLP_final.eqx")
    t0 = time.time()
    print("DED MLP + sPHNN")


    all_features, ts, stats, metadata = load_ded_data(
        DATA_DIR, downsample=1, trajectories=list(range(N_TRAJ)))
    n_nodes = metadata['n_nodes']
    scale = float(stats['temp_scale']); shift = float(stats['temp_shift'])

    all_results = []

    # Split (permutation fixed, N_TRAIN varies per run)
    split_key = jr.key(0); _, subkey = jr.split(split_key)
    perm = jr.permutation(subkey, jnp.arange(N_TRAJ))
    train_idx = perm[:N_TRAIN]; test_idx = perm[N_TRAIN:]
    train_features = all_features[train_idx]; test_features = all_features[test_idx]
    print(f"  Train idx: {np.array(train_idx)} | Test idx: {np.array(test_idx)}")

    # Prepare data for this N_TRAIN
    ts_p1, nodes_p1, seq_p1 = prepare_phase1_data(train_features, ts)
    ts_p2, nodes_p2, seq_p2 = prepare_phase2_data(train_features, ts)

    key = jr.PRNGKey(SEED)
    key, mk, dk = jr.split(key, 3)

    ae = MLPAutoEncoder(n_nodes, LATENT_DIM, mk)
    full_model = MLPsPHNNFullModel(dk, ae)

    if MODEL_PATH.exists():
        print(f"  Loading: {MODEL_PATH}")
        full_model = eqx.tree_deserialise_leaves(MODEL_PATH, full_model)
    else:
        t_run = time.time()

        key, dk1 = jr.split(key)
        full_model, _ = klax.fit(
            full_model, 
            (ts_p1, nodes_p1, seq_p1),
            batch_size=BATCH_P1, 
            batch_axis=(0, 0, 0),
            steps=STEPS_PHASE1, 
            loss_fn=trajectory_loss_combined,
            optimizer=optax.adam(LR_P1),
            history=klax.HistoryCallback(log_every=100), key=dk1)

        key, dk2 = jr.split(key)
        full_model, _ = klax.fit(
            full_model, 
            (ts_p2, nodes_p2, seq_p2),
            batch_size=BATCH_P2, 
            batch_axis=(0, 0, 0),
            steps=STEPS_PHASE2,  
            loss_fn=trajectory_loss_combined,
            optimizer=optax.adam(LR_P2),
            history=klax.HistoryCallback(log_every=100), key=dk2)

        eqx.tree_serialise_leaves(MODEL_PATH, full_model)
        print(f"  Trained in {(time.time()-t_run)/60:.1f} min -> {MODEL_PATH}")

    final = klax.finalize(full_model)
    ae_train, ode_train = compute_mlp_metrics(final, train_features, ts, scale, shift)
    ae_test, ode_test = compute_mlp_metrics(final, test_features, ts, scale, shift)

    result = {'N_TRAIN': N_TRAIN, 'ae_rmse_train': ae_train, 'ae_rmse_test': ae_test,
              'ode_rmse_train': ode_train, 'ode_rmse_test': ode_test}
    all_results.append(result)
    print(f"  AE Train: {ae_train:.2f} K | AE Test: {ae_test:.2f} K")
    print(f"  ODE Train: {ode_train:.2f} K | ODE Test: {ode_test:.2f} K")
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
