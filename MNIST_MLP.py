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

# DATA LOADING

def load_mnist_data(n_samples=10000):
    print("Loading MNIST")
    t0 = time.time()
    mnist = fetch_openml('mnist_784', version=1, parser='auto', as_frame=False)
    X = mnist.data[:n_samples].astype(np.float32) / 255.0
    y = mnist.target[:n_samples].astype(int)
    all_features = jnp.array(X)
    print(f"  Loaded: {n_samples} samples | 784 pixels | {time.time()-t0:.1f}s")
    return all_features, y

# MLP AUTOENCODER

class MLPAutoEncoder(eqx.Module):
    encoder_mlp: eqx.nn.MLP
    decoder_mlp: eqx.nn.MLP
    n_nodes: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)

    def __init__(self, n_nodes, latent_dim, key):
        keys = jr.split(key, 2)
        self.n_nodes = n_nodes
        self.latent_dim = latent_dim
        self.encoder_mlp = eqx.nn.MLP(
            in_size=n_nodes, out_size=latent_dim,
            width_size=512, depth=3,
            activation=jax.nn.relu, key=keys[0])
        self.decoder_mlp = eqx.nn.MLP(
            in_size=latent_dim, out_size=n_nodes,
            width_size=512, depth=3,
            activation=jax.nn.relu,
            final_activation=lambda x: x, key=keys[1])

    def encode(self, pixels):
        return self.encoder_mlp(pixels)

    def decode(self, z):
        return self.decoder_mlp(z)

    def __call__(self, pixels):
        return self.decode(self.encode(pixels))

# LOSS

def recon_loss(model, data, batch_axis):
    """MSE on flat pixel vectors."""
    pixels_batch = data  # shape (B, 784)

    def single_recon(pixels):
        recon = model(pixels)
        return jnp.mean((recon - pixels) ** 2)

    losses = jax.vmap(single_recon)(pixels_batch)
    return jnp.mean(losses)

# EVALUATION

def compute_recon_metrics(model, features):
    """Compute per-sample RMSE."""
    rmses = []
    chunk_size = 500
    for start in range(0, features.shape[0], chunk_size):
        chunk = features[start:start + chunk_size]
        recons = jax.vmap(model)(chunk)
        batch_rmses = jnp.sqrt(jnp.mean((recons - chunk) ** 2, axis=1))
        rmses.extend(np.array(batch_rmses).tolist())
    return float(np.mean(rmses))

def main():
    LATENT_DIM = 32
    STEPS = 10000
    BATCH = 64
    LR = 5e-4
    N_SAMPLES = 10000
    SEED = 42
    MODEL_PATH = Path("mnist_mlp.eqx")
    t0 = time.time()

    all_features, labels = load_mnist_data(N_SAMPLES)

    # Fixed split
    split_key = jr.key(0); _, subkey = jr.split(split_key)
    n_train = int(0.8 * N_SAMPLES)
    perm = jr.permutation(subkey, N_SAMPLES)
    train_features = all_features[perm[:n_train]]
    test_features = all_features[perm[n_train:]]

    key = jr.PRNGKey(SEED); key, mk = jr.split(key)
    model = MLPAutoEncoder(784, LATENT_DIM, mk)

    if MODEL_PATH.exists():
        model = eqx.tree_deserialise_leaves(MODEL_PATH, model)
        print("  Loaded existing model.")
    else:
        key, tk = jr.split(key)
        model, _ = klax.fit(
            model, train_features, batch_size=BATCH, batch_axis=0,
            steps=STEPS, loss_fn=recon_loss, optimizer=optax.adam(LR),
            history=klax.HistoryCallback(log_every=100), key=tk)
        eqx.tree_serialise_leaves(MODEL_PATH, model)

    final = klax.finalize(model)
    train_rmse = compute_recon_metrics(final, train_features)
    test_rmse  = compute_recon_metrics(final, test_features)
    print(f"  Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}")
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
