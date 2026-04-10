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

def build_edges_once(edge_list):
    edge_set, senders, receivers = set(), [], []
    for edge in edge_list:
        i, j = int(edge[0]), int(edge[1])
        if (i, j) not in edge_set and (j, i) not in edge_set:
            senders.extend([i, j]); receivers.extend([j, i]); edge_set.add((i, j))
    return np.array(senders, dtype=np.int32), np.array(receivers, dtype=np.int32)

def load_mesh_edges(mesh_file):
    mesh = pv.read(str(mesh_file))
    edges = mesh.extract_all_edges()
    lines = edges.lines.reshape(-1, 3)
    return lines[:, 1:3], mesh.points

def load_ded_data(data_dir, downsample=1, trajectories=None):
    t_load = time.time()
    raw_data = np.load(data_dir / "data.npz")
    nodes = np.array(raw_data["nodes"], dtype=np.float32)
    temperatures = np.array(raw_data["temp"], dtype=np.float32)
    sources = np.array(raw_data["source"], dtype=np.float32)
    n_trajectories, n_timesteps, n_nodes_orig = temperatures.shape
    if downsample > 1:
        nodes = nodes[::downsample]; temperatures = temperatures[:,:,::downsample]
        sources = sources[:,:,::downsample]
    n_nodes = nodes.shape[0]
    mesh_file = data_dir / "mesh.nas"
    senders, receivers = None, None
    if mesh_file.exists():
        edge_list, _ = load_mesh_edges(mesh_file)
        if downsample > 1:
            valid = set(range(0, n_nodes_orig, downsample))
            mask = np.isin(edge_list[:,0], list(valid)) & np.isin(edge_list[:,1], list(valid))
            edge_list = edge_list[mask] // downsample
        senders, receivers = build_edges_once(edge_list)
    if trajectories is None: trajectories = list(range(n_trajectories))
    temperatures = temperatures[trajectories]; sources = sources[trajectories]
    n_sel = len(trajectories)
    temp_shift = 293.0; temp_std = float(temperatures.std())
    source_max = float(sources.max())
    if source_max < 1e-8: source_max = 1.0
    stats = {'positions': jnp.array(nodes), 'temp_shift': jnp.array(temp_shift),
             'temp_scale': jnp.array(temp_std), 'source_max': jnp.array(source_max), 'n_nodes': n_nodes}
    temp_norm = (temperatures - temp_shift) / temp_std
    source_norm = sources / source_max
    all_features = jnp.array(np.concatenate([temp_norm[:,:,:,None], source_norm[:,:,:,None]], axis=-1).astype(np.float32))
    metadata = {'n_trajectories': n_sel, 'n_timesteps': n_timesteps,
                'n_nodes': n_nodes, 'senders': senders, 'receivers': receivers}
    print(f"Loaded: {n_sel} Traj | {n_nodes} Nodes | {time.time()-t_load:.1f}s")
    return all_features, stats, metadata

def create_geometric_assignment(positions, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(positions)
    S = np.zeros((positions.shape[0], n_clusters), dtype=np.float32)
    for i in range(positions.shape[0]): S[i, labels[i]] = 1.0
    return jnp.array(S)

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

# LAYERS

class GCNLayer(eqx.Module):
    linear: eqx.nn.Linear; adj_norm: jnp.ndarray
    use_activation: bool = eqx.field(static=True)
    def __init__(self, in_features, out_features, adj, key, use_activation=True):
        self.linear = eqx.nn.Linear(in_features, out_features, key=key)
        self.use_activation = use_activation
        adj_tilde = adj + jnp.eye(adj.shape[0])
        degree = jnp.maximum(jnp.sum(adj_tilde, axis=1), 1.0)
        d = jnp.power(degree, -0.5)
        self.adj_norm = adj_tilde * d[:, None] * d[None, :]
    def __call__(self, x):
        h = jnp.matmul(self.adj_norm, x)
        h = jax.vmap(self.linear)(h)
        return jnp.tanh(h) if self.use_activation else h

class PoolingLayer(eqx.Module):
    S: jnp.ndarray; k: int = eqx.field(static=True)
    def __init__(self, S_matrix): self.S = S_matrix; self.k = S_matrix.shape[1]
    def pool_nodes(self, nodes):
        sizes = jnp.sum(self.S, axis=0)
        S_norm = self.S / jnp.where(sizes > 0, sizes, 1.0)[None, :]
        return jnp.matmul(S_norm.T, nodes)

class UnpoolingLayer(eqx.Module):
    def __call__(self, nodes_pooled, S): return jnp.matmul(S, nodes_pooled)

# GAE AUTOENCODER

class GAEAutoEncoder(eqx.Module):
    enc_gcn_l0: GCNLayer; pool1: PoolingLayer; enc_gcn_l1: GCNLayer
    pool2: PoolingLayer; enc_gcn_l2: GCNLayer; enc_mlp_to_latent: eqx.nn.MLP
    dec_mlp_from_latent: eqx.nn.MLP; dec_gcn_l2: GCNLayer; unpool2: UnpoolingLayer
    dec_gcn_l1: GCNLayer; unpool1: UnpoolingLayer; dec_gcn_l0: GCNLayer
    latent_dim: int = eqx.field(static=True)
    pool2_nodes: int = eqx.field(static=True)
    n_nodes: int = eqx.field(static=True)

    def __init__(self, in_features, latent_dim, S1, S2, senders, receivers, n_nodes, key):
        keys = jax.random.split(key, 10)
        self.latent_dim = latent_dim; self.pool2_nodes = S2.shape[1]; self.n_nodes = n_nodes
        adj_l0 = jnp.zeros((n_nodes, n_nodes)).at[senders, receivers].set(1.0)
        adj_l1 = jnp.matmul(jnp.matmul(S1.T, adj_l0), S1)
        adj_l2 = jnp.matmul(jnp.matmul(S2.T, adj_l1), S2)
        self.enc_gcn_l0 = GCNLayer(in_features, 8, adj_l0, keys[0], True)
        self.pool1 = PoolingLayer(S1)
        self.enc_gcn_l1 = GCNLayer(8, 16, adj_l1, keys[1], True)
        self.pool2 = PoolingLayer(S2)
        self.enc_gcn_l2 = GCNLayer(16, 32, adj_l2, keys[2], True)
        self.enc_mlp_to_latent = eqx.nn.MLP(self.pool2_nodes * 32, latent_dim, 256, 2, key=keys[3])
        self.dec_mlp_from_latent = eqx.nn.MLP(latent_dim, self.pool2_nodes * 32, 256, 2, key=keys[4])
        self.dec_gcn_l2 = GCNLayer(32, 16, adj_l2, keys[5], True)
        self.unpool2 = UnpoolingLayer()
        self.dec_gcn_l1 = GCNLayer(16, 8, adj_l1, keys[6], True)
        self.unpool1 = UnpoolingLayer()
        self.dec_gcn_l0 = GCNLayer(8, 1, adj_l0, keys[7], False)

    def _encode_raw(self, x):
        x = self.enc_gcn_l0(x); x = self.pool1.pool_nodes(x)
        x = self.enc_gcn_l1(x); x = self.pool2.pool_nodes(x)
        x = self.enc_gcn_l2(x); return self.enc_mlp_to_latent(x.reshape(-1))

    def _decode_raw(self, z):
        x = self.dec_mlp_from_latent(z).reshape(self.pool2_nodes, 32)
        x = self.dec_gcn_l2(x); x = self.unpool2(x, self.pool2.S)
        x = self.dec_gcn_l1(x); x = self.unpool1(x, self.pool1.S)
        return self.dec_gcn_l0(x)

    def encode(self, nodes):
        temp_only = nodes[:, 0:1]
        return self._encode_raw(temp_only) - self._encode_raw(jnp.zeros_like(temp_only))

    def decode(self, z):
        return self._decode_raw(z) - self._decode_raw(jnp.zeros_like(z))

    def encode_with_cached_zero(self, nodes, z_0):
        temp_only = nodes[:, 0:1]
        return self._encode_raw(temp_only) - z_0

    def decode_with_cached_zero(self, z, t_0):
        return self._decode_raw(z) - t_0

    def __call__(self, nodes):
        return self.decode(self.encode(nodes))


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
        keys = jr.split(key, 6)
        ficnn = klax.nn.FICNN(state_size, "scalar", [16, 16],
                               weight_init=variance_scaling(1, "fan_avg", "truncated_normal"), key=keys[0])
        self.deriv_model = ISPHS(
            ConvexLyapunov(ficnn, state_size=state_size, key=keys[1]),
            klax.nn.ConstantSkewSymmetricMatrix(state_size, init=variance_scaling(1, "fan_avg", "truncated_normal"), key=keys[2]),
            klax.nn.ConstantSPDMatrix(state_size, epsilon=0.0, init=variance_scaling(1, "fan_avg", "truncated_normal"), key=keys[3]),
            klax.nn.ConstantMatrix((state_size, laser_latent_size), init=variance_scaling(1, "fan_avg", "truncated_normal"), key=keys[4]))
        self.solver = ODESolver(self.deriv_model)
        self.laser_encoder = LaserInputEncoder(keys[5], n_nodes=n_nodes, output_dim=laser_latent_size)
    def __call__(self, ts, z0, source_sequence):
        return self.solver(ts, z0, jax.vmap(self.laser_encoder)(source_sequence))

class GAEsPHNNFullModel(eqx.Module):
    ae: GAEAutoEncoder; dynamics: sPHNNDynamics
    def __init__(self, key, ae_architecture):
        self.ae = ae_architecture
        self.dynamics = sPHNNDynamics(key, state_size=ae_architecture.latent_dim,
                                     laser_latent_size=8, n_nodes=ae_architecture.n_nodes)

# LOSS

RECON_WEIGHT = 0.5
def trajectory_loss_combined(model, data, batch_axis):
    ts_batch, nodes_init, nodes_seq = data
    temp_zero = jnp.zeros((model.ae.n_nodes, 1))
    z_0 = model.ae._encode_raw(temp_zero)
    t_0 = model.ae._decode_raw(jnp.zeros(model.ae.latent_dim))

    def predict_single(ts_i, init_nodes, seq_nodes):
        zs_true = jax.vmap(lambda n: model.ae.encode_with_cached_zero(n, z_0))(seq_nodes)
        z0 = zs_true[0]
        recons = jax.vmap(lambda z: model.ae.decode_with_cached_zero(z, t_0))(zs_true)
        target_temp = seq_nodes[:, :, 0:1]
        recon_loss = jnp.mean((recons - target_temp) ** 2)
        source_seq = seq_nodes[:, :, 1]
        zs_pred = model.dynamics(ts_i, z0, source_seq)
        preds = jax.vmap(lambda z: model.ae.decode_with_cached_zero(z, t_0))(zs_pred)
        dyn_loss = jnp.mean((preds - target_temp) ** 2)
        return dyn_loss + RECON_WEIGHT * recon_loss

    losses = jax.vmap(predict_single)(ts_batch, nodes_init, nodes_seq)
    return jnp.mean(losses)

# EVALUATION

def compute_gae_metrics(model, features, ts, scale, shift):
    ae_rmses, ode_rmses = [], []
    for ti in range(features.shape[0]):
        traj = features[ti]
        zs = jax.vmap(model.ae.encode)(traj)
        recon = jax.vmap(lambda z: model.ae.decode(z).squeeze())(zs)
        recon_k = np.array(recon) * scale + shift
        true_k = np.array(traj[:, :, 0]) * scale + shift
        ae_rmses.append(float(np.sqrt(np.mean((recon_k - true_k)**2))))
        z0 = model.ae.encode(traj[0])
        zs_pred = model.dynamics(ts, z0, traj[:, :, 1])
        pred = jax.vmap(lambda z: model.ae.decode(z).squeeze())(zs_pred)
        pred_k = np.array(pred) * scale + shift
        ode_rmses.append(float(np.sqrt(np.mean((pred_k - true_k)**2))))
    return float(np.mean(ae_rmses)), float(np.mean(ode_rmses))

# MAIN

def main():
    SEED_POOL    = 42
    STEPS_PHASE1 = 10000
    STEPS_PHASE2 = 4000
    BATCH_P1     = 128
    BATCH_P2     = 4
    LR_P1        = 4e-4
    LR_P2        = 4e-4
    N_TRAJ       = 25
    N_TRAIN      = 4
    LATENT_DIM   = 16
    SEED         = 42
    MODEL_PATH    = Path("DED_GAE")
    DATA_DIR     = Path("ded_data")
    t0 = time.time()

    print(f"DED GAE + sPHNN")

    all_features, stats, metadata = load_ded_data(
        DATA_DIR, downsample=1, trajectories=list(range(N_TRAJ)))
    n_nodes = metadata['n_nodes']
    senders = metadata['senders']; receivers = metadata['receivers']
    positions = np.array(stats['positions'])
    raw_data = np.load(DATA_DIR / "data.npz")
    ts = jnp.array(raw_data["ts"])
    scale = float(stats['temp_scale']); shift = float(stats['temp_shift'])

    # Split
    split_key = jr.key(0); _, subkey = jr.split(split_key)
    perm = jr.permutation(subkey, jnp.arange(N_TRAJ))

    # Pooling
    k1, k2 = 400, 64
    S1 = create_geometric_assignment(positions, n_clusters=k1)
    cs1 = np.sum(S1, axis=0); S1n = S1 / np.where(cs1 > 0, cs1, 1.0)[None, :]
    pos_p1 = S1n.T @ positions
    labels2 = KMeans(k2, random_state=SEED_POOL, n_init=10).fit_predict(pos_p1)
    S2 = np.zeros((k1, k2), dtype=np.float32)
    for i in range(k1): S2[i, labels2[i]] = 1.0
    S2 = jnp.array(S2)

    all_results = []

    train_idx = perm[:N_TRAIN]; test_idx = perm[N_TRAIN:]
    train_features = all_features[train_idx]; test_features = all_features[test_idx]
    print(f"  Train idx: {np.array(train_idx)} | Test idx: {np.array(test_idx)}")

    ts_p1, nodes_p1, seq_p1 = prepare_phase1_data(train_features, ts)
    ts_p2, nodes_p2, seq_p2 = prepare_phase2_data(train_features, ts)

    key = jr.PRNGKey(SEED)
    key, mk, dk = jr.split(key, 3)

    ae = GAEAutoEncoder(1, LATENT_DIM, S1, S2, senders, receivers, int(n_nodes), mk)
    full_model = GAEsPHNNFullModel(dk, ae)

    if MODEL_PATH.exists():
        print(f"  Loading: {MODEL_PATH}")
        full_model = eqx.tree_deserialise_leaves(MODEL_PATH, full_model)
    else:
        t_run = time.time()

        # Phase 1
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

        # Phase 2
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

    # Evaluate
    final = klax.finalize(full_model)
    ae_train, ode_train = compute_gae_metrics(final, train_features, ts, scale, shift)
    ae_test, ode_test = compute_gae_metrics(final, test_features, ts, scale, shift)

    result = {'N_TRAIN': N_TRAIN, 'ae_rmse_train': ae_train, 'ae_rmse_test': ae_test,
              'ode_rmse_train': ode_train, 'ode_rmse_test': ode_test}
    all_results.append(result)
    print(f"  AE Train: {ae_train:.2f} K | AE Test: {ae_test:.2f} K")
    print(f"  ODE Train: {ode_train:.2f} K | ODE Test: {ode_test:.2f} K")
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
