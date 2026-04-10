import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
import matplotlib.pyplot as plt
from pathlib import Path
import time
from jax import random as jr
import klax
from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree

# DATA LOADING
def build_edges_grid(k_neighbors=8):
    """Build k-NN edges for 28x28 grid (same logic as DED mesh edges)."""
    x, y = np.meshgrid(np.arange(28), np.arange(28))
    positions = np.stack([x.ravel(), y.ravel()], axis=1).astype(np.float32)

    tree = cKDTree(positions)
    _, indices = tree.query(positions, k=k_neighbors + 1)

    edge_set, senders, receivers = set(), [], []
    for i in range(784):
        for j in indices[i, 1:]:
            j = int(j)
            if (i, j) not in edge_set and (j, i) not in edge_set:
                senders.extend([i, j])
                receivers.extend([j, i])
                edge_set.add((i, j))

    return (np.array(senders, dtype=np.int32),
            np.array(receivers, dtype=np.int32),
            positions / 27.0)


def load_mnist_data(n_samples=10000, k_neighbors=8):
    """Load MNIST as node feature arrays (no jraph). Returns same format as DED."""
    print(f"Loading MNIST ({n_samples} samples, k={k_neighbors} neighbors)...")
    t0 = time.time()

    mnist = fetch_openml('mnist_784', version=1, parser='auto', as_frame=False)
    X = mnist.data[:n_samples].astype(np.float32) / 255.0
    y = mnist.target[:n_samples].astype(int)

    senders, receivers, positions = build_edges_grid(k_neighbors)

    # Node features: [pos_x, pos_y, pixel] — shape (N_samples, 784, 3)
    node_features = np.zeros((n_samples, 784, 3), dtype=np.float32)
    node_features[:, :, 0] = np.tile(positions[:, 0], (n_samples, 1))
    node_features[:, :, 1] = np.tile(positions[:, 1], (n_samples, 1))
    node_features[:, :, 2] = X

    all_features = jnp.array(node_features)

    print(f"  Loaded: {n_samples} samples | 784 nodes | {len(senders)} edges | {time.time()-t0:.1f}s")
    return all_features, y, senders, receivers, positions


def create_geometric_assignment(positions, n_clusters):
    """Identical to DED version."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(positions)
    S = np.zeros((positions.shape[0], n_clusters), dtype=np.float32)
    for i in range(positions.shape[0]):
        S[i, labels[i]] = 1.0
    return jnp.array(S)

# LAYERS
class GCNLayer(eqx.Module):
    linear: eqx.nn.Linear
    adj_norm: jnp.ndarray
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
        return jax.nn.relu(h) if self.use_activation else h


class PoolingLayer(eqx.Module):
    S: jnp.ndarray
    k: int = eqx.field(static=True)

    def __init__(self, S_matrix):
        self.S = S_matrix
        self.k = S_matrix.shape[1]

    def pool_nodes(self, nodes):
        sizes = jnp.sum(self.S, axis=0)
        S_norm = self.S / jnp.where(sizes > 0, sizes, 1.0)[None, :]
        return jnp.matmul(S_norm.T, nodes)


class UnpoolingLayer(eqx.Module):
    def __call__(self, nodes_pooled, S):
        return jnp.matmul(S, nodes_pooled)


# GAE AUTOENCODER
class GAEAutoEncoder(eqx.Module):
    enc_gcn_l0: GCNLayer
    pool1: PoolingLayer
    enc_gcn_l1: GCNLayer
    pool2: PoolingLayer
    enc_gcn_l2: GCNLayer
    enc_mlp_to_latent: eqx.nn.MLP
    dec_mlp_from_latent: eqx.nn.MLP
    dec_gcn_l2: GCNLayer
    unpool2: UnpoolingLayer
    dec_gcn_l1: GCNLayer
    unpool1: UnpoolingLayer
    dec_gcn_l0: GCNLayer
    latent_dim: int = eqx.field(static=True)
    pool2_nodes: int = eqx.field(static=True)
    n_nodes: int = eqx.field(static=True)

    def __init__(self, in_features, latent_dim, S1, S2, senders, receivers, n_nodes, key):
        keys = jax.random.split(key, 10)
        self.latent_dim = latent_dim
        self.pool2_nodes = S2.shape[1]
        self.n_nodes = n_nodes

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
        self.dec_gcn_l0 = GCNLayer(8, in_features, adj_l0, keys[7], False)

    def encode(self, nodes):
        x = self.enc_gcn_l0(nodes)
        x = self.pool1.pool_nodes(x)
        x = self.enc_gcn_l1(x)
        x = self.pool2.pool_nodes(x)
        x = self.enc_gcn_l2(x)
        return self.enc_mlp_to_latent(x.reshape(-1))

    def decode(self, z):
        x = self.dec_mlp_from_latent(z).reshape(self.pool2_nodes, 32)
        x = self.dec_gcn_l2(x)
        x = self.unpool2(x, self.pool2.S)
        x = self.dec_gcn_l1(x)
        x = self.unpool1(x, self.pool1.S)
        return self.dec_gcn_l0(x)

    def __call__(self, nodes):
        return self.decode(self.encode(nodes))


# LOSS

def recon_loss(model, data, batch_axis):
    nodes_batch = data
    def single_recon(nodes):
        recon = model(nodes)
        return jnp.mean((recon[:, 2] - nodes[:, 2]) ** 2)

    losses = jax.vmap(single_recon)(nodes_batch)
    return jnp.mean(losses)

# EVALUATION

def compute_recon_metrics(model, features):
    """Compute per-sample RMSE on pixel channel (index 2)."""
    rmses = []
    chunk_size = 100
    for start in range(0, features.shape[0], chunk_size):
        chunk = features[start:start + chunk_size]
        recons = jax.vmap(model)(chunk)
        for i in range(chunk.shape[0]):
            rmse = float(jnp.sqrt(jnp.mean((recons[i, :, 2] - chunk[i, :, 2]) ** 2)))
            rmses.append(rmse)
    return float(np.mean(rmses))

# MAIN

def main():
    SEED = 42
    LATENT_DIM = 32
    STEPS = 10000
    BATCH = 64
    LR = 1e-3
    N_SAMPLES = 10000
    K_NEIGHBORS = 8
    k1, k2 = 200, 40
    MODEL_PATH = Path("MNIST_GAE_final.eqx")
    t0 = time.time()
    print("MNIST GAE Autoencoder")
    print(f"  d={LATENT_DIM}, LR={LR}, BS={BATCH}, k={K_NEIGHBORS}, Pool={k1}/{k2}")
    # Load data
    all_features, labels, senders, receivers, positions = load_mnist_data(N_SAMPLES, K_NEIGHBORS)

    # Train/test split
    n_train = int(0.8 * N_SAMPLES)
    key = jr.PRNGKey(SEED)
    key, split_key = jr.split(key)
    perm = jr.permutation(split_key, N_SAMPLES)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    train_features = all_features[train_idx]
    test_features = all_features[test_idx]
    train_labels = labels[np.array(train_idx)]
    test_labels = labels[np.array(test_idx)]
    print(f"Split: {n_train} train / {N_SAMPLES - n_train} test")

    S1 = create_geometric_assignment(positions, n_clusters=k1)
    cs1 = np.sum(S1, axis=0)
    S1n = S1 / np.where(cs1 > 0, cs1, 1.0)[None, :]
    pos_p1 = S1n.T @ positions

    labels_2 = KMeans(k2, random_state=42, n_init=10).fit_predict(pos_p1)
    S2 = np.zeros((k1, k2), dtype=np.float32)
    for i in range(k1):
        S2[i, labels_2[i]] = 1.0
    S2 = jnp.array(S2)

    # Model
    key, mk = jr.split(key)
    model = GAEAutoEncoder(3, LATENT_DIM, S1, S2, senders, receivers, 784, mk)

    params = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))
    print(f"Model: {params:,} parameters | Latent={LATENT_DIM}")

    if MODEL_PATH.exists():
        print(f"Loading: {MODEL_PATH}")
        model = eqx.tree_deserialise_leaves(MODEL_PATH, model)
    else:
        print(f"Training {STEPS} steps...")

        key, tk = jr.split(key)
        model, history = klax.fit(
            model,
            train_features,
            batch_size=BATCH,
            batch_axis=0,
            steps=STEPS,
            loss_fn=recon_loss,
            optimizer=optax.adam(LR),
            history=klax.HistoryCallback(log_every=100),
            key=tk)

        eqx.tree_serialise_leaves(MODEL_PATH, model)
        print(f"Saved: {MODEL_PATH}")

    # Evaluate
    final = klax.finalize(model)
    train_rmse = compute_recon_metrics(final, train_features)
    test_rmse = compute_recon_metrics(final, test_features)
    print(f"\nResults:")
    print(f"  Train RMSE: {train_rmse:.4f}")
    print(f"  Test  RMSE: {test_rmse:.4f}")
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
