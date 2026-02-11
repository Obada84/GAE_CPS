import cyberplotstyle as cps
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
import jax.tree_util as jtu
import pyvista as pv
from dynax import ISPHS, ConvexLyapunov, ODESolver
from jax.nn.initializers import variance_scaling
from matplotlib.lines import Line2D

# DATA LOADING


def load_ded_data(data_dir, downsample=1, trajectories=None):  
    raw_data = np.load(data_dir / "data.npz")
    nodes = np.array(raw_data["nodes"])
    temperatures = np.array(raw_data["temp"])
    sources = np.array(raw_data["source"])
    ts = np.array(raw_data["ts"])
    
    n_trajectories, n_timesteps, n_nodes_orig = temperatures.shape
    
    if downsample > 1:
        nodes = nodes[::downsample]
        temperatures = temperatures[:, :, ::downsample]
        sources = sources[:, :, ::downsample]
    
    if trajectories is None:
        trajectories = list(range(n_trajectories))

    nodes_data = nodes
    temp_data = temperatures[trajectories]
    source_data = sources[trajectories]
    
    n_nodes = nodes_data.shape[0]
    
    # Position normalization
    pos_mean = nodes_data.mean(axis=0)
    pos_std = np.where(nodes_data.std(axis=0) > 1e-8, nodes_data.std(axis=0), 1.0)
    
    # TEMPERATURE: POD-STYLE (shift=293, scale=std)
    temp_shift = 293.0  # temperature → 0
    temp_scale = float(temp_data.std())
    
    # Source normalization
    source_max = float(source_data.max())
    if source_max < 1e-8:
        source_max = 1.0
    
    stats = {
        'pos_mean': jnp.array(pos_mean),
        'pos_std': jnp.array(pos_std),
        'temp_shift': jnp.array(temp_shift),
        'temp_scale': jnp.array(temp_scale),
        'source_max': jnp.array(source_max),
        'temp_min': float(temp_data.min()),
        'temp_max': float(temp_data.max()),
        'n_nodes': n_nodes
    }
    
    # Normalize
    pos_norm = (nodes_data - stats['pos_mean']) / stats['pos_std']
    temp_norm = (temp_data - stats['temp_shift']) / stats['temp_scale']
    source_norm = source_data / stats['source_max']
    

    all_features = []
    for i in range(len(trajectories)):
        traj_features = []
        for t in range(n_timesteps):
            # [n_nodes, 5]: [x, y, z, temp, source]
            features = jnp.concatenate([
                jnp.tile(pos_norm, (1, 1)),
                temp_norm[i, t:t+1].T,
                source_norm[i, t:t+1].T
            ], axis=1)
            traj_features.append(features)
        all_features.append(jnp.stack(traj_features))
    
    all_features = jnp.stack(all_features)
    
    metadata = {
        'n_trajectories': len(trajectories),
        'n_timesteps': n_timesteps,
        'n_nodes': n_nodes,
        'trajectory_indices': trajectories,
        'feature_names': ['x', 'y', 'z', 'temp', 'source']
    }
    
    return all_features, ts, stats, metadata


# MLP AUTOENCODER


class MLPAutoEncoder(eqx.Module):
    encoder_mlp: eqx.nn.MLP
    decoder_mlp: eqx.nn.MLP
    
    input_dim: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)
    
    def __init__(self, n_nodes, latent_dim, key):
        keys = jr.split(key, 2)
        
        self.input_dim = n_nodes * 5 
        self.latent_dim = latent_dim
        self.output_dim = n_nodes
        
        # Encoder
        self.encoder_mlp = eqx.nn.MLP(
            in_size=self.input_dim,
            out_size=latent_dim,
            width_size=512,
            depth=3,
            activation=jax.nn.leaky_relu,
            key=keys[0]
        )
        
        # Decoder
        self.decoder_mlp = eqx.nn.MLP(
            in_size=latent_dim,
            out_size=self.output_dim,
            width_size=512,
            depth=3,
            activation=jax.nn.leaky_relu,
            final_activation=lambda x: x, 
            key=keys[1]
        )
    
    def encode(self, features):
        x_flat = features.reshape(-1)
        z = self.encoder_mlp(x_flat)
        return z
    
    def decode(self, z):
        temp_pred = self.decoder_mlp(z)
        return temp_pred.reshape(-1, 1)
    
    def __call__(self, features):
        z = self.encode(features)
        temp_pred = self.decode(z)
        return temp_pred


# LASER ENCODER

class LaserInputEncoder(eqx.Module):
    mlp: eqx.nn.MLP
    output_dim: int = eqx.field(static=True)
    
    def __init__(self, key, output_dim=8):
        self.output_dim = output_dim
        self.mlp = eqx.nn.MLP(4, output_dim, 16, 2, activation=jax.nn.leaky_relu, key=key)
    
    def __call__(self, features):
        positions = features[:, :3]
        source = features[:, 4:5]
        
        max_idx = jnp.argmax(source)
        laser_pos = positions[max_idx]
        laser_intensity = source[max_idx, 0]
        
        laser_features = jnp.concatenate([laser_pos, jnp.array([laser_intensity])])
        u = self.mlp(laser_features)
        
        return u


# sPHNN DYNAMICS

class sPHNNDynamics(eqx.Module):
    deriv_model: ISPHS
    solver: ODESolver
    laser_encoder: LaserInputEncoder
    
    def __init__(self, key, state_size=32, laser_latent_size=8):
        keys = jr.split(key, 6)
        
        ficnn = klax.nn.FICNN(
            in_size=state_size,
            out_size="scalar",
            width_sizes=[16, 16],
            weight_init=variance_scaling(1, "fan_avg", "truncated_normal"),
            key=keys[0],
        )
        hamiltonian = ConvexLyapunov(ficnn, state_size=state_size, key=keys[1])
        
        structure_matrix = klax.nn.ConstantSkewSymmetricMatrix(
            state_size,
            init=variance_scaling(1, "fan_avg", "truncated_normal"),
            key=keys[2]
        )
        
        dissipation_matrix = klax.nn.ConstantSPDMatrix(
            state_size,
            epsilon=0.2,
            init=variance_scaling(1.0, "fan_avg", "truncated_normal"),
            key=keys[3]
        )
        
        input_matrix = klax.nn.ConstantMatrix(
            (state_size, laser_latent_size),
            init=variance_scaling(1, "fan_avg", "truncated_normal"),
            key=keys[4]
        )
        
        self.deriv_model = ISPHS(hamiltonian, structure_matrix, dissipation_matrix, input_matrix)
        self.solver = ODESolver(self.deriv_model)
        self.laser_encoder = LaserInputEncoder(keys[5], output_dim=laser_latent_size)
    
    def __call__(self, ts, z0, features_sequence):
        us = jax.vmap(self.laser_encoder)(features_sequence)
        return self.solver(ts, z0, us)

# COMBINED MODEL

class MLPsPHNNFullModel(eqx.Module):
    encoder: MLPAutoEncoder
    decoder: MLPAutoEncoder
    dynamics: sPHNNDynamics
    
    def __init__(self, key, n_nodes, latent_dim=32):
        keys = jr.split(key, 2)
        
        ae_model = MLPAutoEncoder(n_nodes, latent_dim, keys[0])
        self.encoder = ae_model
        self.decoder = ae_model
        self.dynamics = sPHNNDynamics(keys[1], state_size=latent_dim, laser_latent_size=8)
    
    def predict_trajectory(self, ts, features_sequence):
        z0 = self.encoder.encode(features_sequence[0])
        zs = self.dynamics(ts, z0, features_sequence)
        
        temps_pred = jax.vmap(self.decoder.decode)(zs)
        return temps_pred


# DATA PREPARATION

def to_step_pairs(features_traj, ts):
    n_timesteps = features_traj.shape[0]
    
    pairs = []
    for i in range(n_timesteps - 1):
        pairs.append((features_traj[i], features_traj[i+1]))
    
    return pairs, ts[:-1], ts[1:]


# AE WARMUP LOSS

def ae_reconstruction_loss(model, data, batch_axis):
    features, _ = data
    
    def single_recon_loss(feat):
        pred = model(feat)
        true = feat[:, 3:4] 
        return jnp.mean((pred - true)**2)
    
    return jnp.mean(jax.vmap(single_recon_loss)(features))

# TRAJECTORY LOSS 

def trajectory_loss_phase1(model, data, batch_axis):

    ts_pairs, features_t, features_next = data
    
    def single_step_loss(t_pair, feat_t, feat_next):
        z_t = jax.lax.stop_gradient(model.encoder.encode(feat_t))
        z_true_next = jax.lax.stop_gradient(model.encoder.encode(feat_next))
        
        dt = t_pair[1] - t_pair[0]
        ts_local = jnp.array([0.0, dt])
        features_seq = jnp.stack([feat_t, feat_next])
        z_pred = model.dynamics(ts_local, z_t, features_seq)[-1]

        return jnp.mean((z_pred - z_true_next)**2)
    
    losses = jax.vmap(single_step_loss)(ts_pairs, features_t, features_next)
    return jnp.mean(losses)


def trajectory_loss_phase2(model, data, batch_axis):
    features_sequences, ts_traj = data
    
    def single_traj_loss(features_traj):
        z_true = jax.vmap(
            lambda f: jax.lax.stop_gradient(model.encoder.encode(f))
        )(features_traj)
        
        z0 = z_true[0]
        z_pred = model.dynamics(ts_traj, z0, features_traj)
        
        return jnp.mean((z_pred - z_true)**2)
    
    return jnp.mean(jax.vmap(single_traj_loss)(features_sequences))


# BATCHERS

def ae_batcher(data, batch_size=64, batch_axis=None, convert_to_numpy=True, *, key):
    features, targets = data
    n_samples = len(features)
    
    while True:
        key, subkey = jax.random.split(key)
        indices = jax.random.permutation(subkey, n_samples)
        
        for i in range(0, n_samples - batch_size + 1, batch_size):
            idx = indices[i : i + batch_size]
            
            batch_features = jnp.stack([features[int(j)] for j in idx])
            batch_targets = jnp.stack([targets[int(j)] for j in idx])
            
            yield (batch_features, batch_targets)


def step_batcher_phase1(data, batch_size=8, batch_axis=None, convert_to_numpy=True, *, key):
    ts_pairs, features_t_list, features_next_list = data
    n_samples = len(features_t_list)
    
    while True:
        key, subkey = jax.random.split(key)
        indices = jax.random.permutation(subkey, n_samples)
        
        for i in range(0, n_samples - batch_size + 1, batch_size):
            idx = indices[i : i + batch_size]
            
            batch_ts = jnp.stack([ts_pairs[int(j)] for j in idx])
            batch_ft = jnp.stack([features_t_list[int(j)] for j in idx])
            batch_fn = jnp.stack([features_next_list[int(j)] for j in idx])
            
            yield (batch_ts, batch_ft, batch_fn)


def trajectory_batcher_phase2(data, batch_size=8, batch_axis=None, convert_to_numpy=True, *, key):
    features_sequences, ts_traj = data
    n_trajectories = features_sequences.shape[0]
    
    while True:
        key, subkey = jax.random.split(key)
        indices = jax.random.permutation(subkey, n_trajectories)
        
        for i in range(0, n_trajectories - batch_size + 1, batch_size):
            idx = indices[i : i + batch_size]
            
            batch_seqs = features_sequences[idx]
            
            yield (batch_seqs, ts_traj)


# VISUALIZATION

def plot_training_curves(hist_ae, hist_phase1, hist_phase2):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    axes[0].plot(hist_ae.loss, color='blue', linewidth=2)
    axes[0].set_title('Phase 0: AE Warmup', fontsize=12)
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(hist_phase1.loss, color='orange', linewidth=2)
    axes[1].set_title('Phase 1: Single-Step', fontsize=12)
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Loss')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(hist_phase2.loss, color='red', linewidth=2)
    axes[2].set_title('Phase 2: Full Trajectory', fontsize=12)
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('Loss')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_mlp.png', dpi=300)
    plt.show()

def plot_1_first_10_latent_variables_training(model, train_features, ts, traj_idx=0):

    traj = train_features[traj_idx]
    zs = jax.vmap(model.encode)(traj)
    zs_np = np.array(zs)
    
    plt.figure(figsize=(10, 6))
    
    colors = plt.cm.tab10(range(10))
    for i in range(min(10, zs_np.shape[1])):
        plt.plot(ts, zs_np[:, i], label=f'Mode {i}', color=colors[i], linewidth=1.5)
    
    plt.xlabel('Time [s]')
    plt.ylabel('Latent Variable')
    plt.title('First 10 Latent Variables (MLP)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('mlp_plot_1_first_10_latent_variables.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_2_first_10_field_values_reconstruction(model, train_features, stats, ts, traj_idx=0):
    
    traj = train_features[traj_idx] 
    
    zs = jax.vmap(model.encode)(traj)
    reconstructed = jax.vmap(model.decode)(zs)
    
    true_temps = traj[:, :, 3] * stats['temp_scale'] + stats['temp_shift']
    recon_temps = reconstructed.squeeze(-1) * stats['temp_scale'] + stats['temp_shift']
    
    rmse = jnp.sqrt(jnp.mean((true_temps - recon_temps) ** 2))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    lines = ax.plot(ts, true_temps[:, :10], linewidth=2)
    
    for n, line in enumerate(lines):
        ax.plot(
            ts,
            recon_temps[:, n],
            '--',
            color=cps.scale_hls(line.get_color(), lightness=0.8),
            linewidth=2
        )
    
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Temperature [K]')
    ax.set_title('First 10 field values reconstruction (MLP)')
    
    legend_elements = [
        Line2D([0], [0], color='grey', linestyle='-', linewidth=2, label='True'),
        Line2D([0], [0], color='grey', linestyle='--', linewidth=2, label='Reconstructed'),
    ]
    ax.legend(handles=legend_elements)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mlp_plot_2_first_10_field_values_reconstruction.png', dpi=300)
    plt.show()
    
    print(f"   Compression RMSE: {rmse:.3e} K")
    return rmse


def plot_3_first_10_latent_variables_test_set(model, test_features, ts, traj_idx=0):    
    test_traj = test_features[traj_idx]
    
    z_true = jax.vmap(model.encoder.encode)(test_traj)
    
    z0 = model.encoder.encode(test_traj[0])
    z_pred = model.dynamics(ts, z0, test_traj)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab10(range(10))
    for i in range(min(10, z_true.shape[1])):
        ax.plot(ts, z_true[:, i], '--', linewidth=2, color=colors[i], 
                alpha=0.7, label=f'Mode {i}')
        ax.plot(ts, z_pred[:, i], linewidth=2, color=colors[i])
    
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Latent Variable')
    ax.set_title('First 10 Latent Variables (Test Set - MLP)')
    
    legend_elements = [
        Line2D([0], [0], color='grey', linestyle='--', linewidth=2, label='True'),
        Line2D([0], [0], color='grey', linestyle='-', linewidth=2, label='Pred'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mlp_plot_3_first_10_latent_test_set.png', dpi=300)
    plt.show()
    


def plot_4_relaxation_prediction_u0(model, test_features, stats, ts, traj_idx=0, start_step=50):
    
    test_traj = test_features[traj_idx]
    features_relax = test_traj[start_step]
    
    ts_relax = jnp.linspace(0, 100, 1000)
    z0_relax = model.encoder.encode(features_relax)
    
    n_nodes = features_relax.shape[0]
    features_relax_seq = jnp.zeros((len(ts_relax), n_nodes, 5))
    features_relax_seq = features_relax_seq.at[:, :, :3].set(features_relax[:, :3])
    
    z_relax = model.dynamics(ts_relax, z0_relax, features_relax_seq)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab10(range(10))
    for i in range(min(10, z_relax.shape[1])):
        ax.plot(ts_relax, z_relax[:, i], linewidth=2, color=colors[i])
    
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Latent Variable')
    ax.set_title('Relaxation Prediction (u=0 - MLP)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mlp_plot_4_relaxation_prediction_u0.png', dpi=300)
    plt.show()
    

# MAIN

def main():
    
    SEED = 42
    LATENT = 32
    
    STEPS_AE_WARMUP = 5000  
    STEPS_PHASE1 = 10000    
    STEPS_PHASE2 = 3000    
    BATCH_SIZE = 8
    LR_AE = 5e-4        
    LR_DYN1 = 7e-4
    LR_DYN2 = 7e-4         
    
    N_TRAJECTORIES_TO_USE = 25
    MODEL_DIR = Path("mlp_models")
    DATA_DIR = Path("ded_data")
    
    start_total = time.time()

    all_features, ts, stats, metadata = load_ded_data(
        DATA_DIR, 
        downsample=1,
        trajectories=list(range(N_TRAJECTORIES_TO_USE))
    )
    
    n_nodes = metadata['n_nodes']
    
    # Split train/test
    train_features = all_features[:20]
    test_features = all_features[20:]

    MODEL_PATH = MODEL_DIR / "sPHNN_MLP_exact_pod.eqx"
    HIST_PATH = MODEL_DIR / "sPHNN_MLP_exact_pod_hist.pkl"

    # Initialize model
    key = jr.PRNGKey(SEED)
    key, mk = jr.split(key)
    
    model = MLPsPHNNFullModel(mk, n_nodes=n_nodes, latent_dim=LATENT)

    if MODEL_PATH.exists():
        print(f"\n LOADING MLP MODEL: {MODEL_PATH}")
        model = eqx.tree_deserialise_leaves(MODEL_PATH, model)
        hist_phase2 = klax.HistoryCallback.load(HIST_PATH)
    else:
        print("\n NO MODEL FOUND: Starting Training...")
        
        # PHASE 0: AE WARMUP (wie POD basis fitting)

        print("\n" + "="*80)
        print("PHASE 0: AE WARMUP")
        print("="*80)
        
        train_samples = []
        for traj in train_features:
            for t in range(traj.shape[0]):
                train_samples.append(traj[t])
        
        train_samples = jnp.stack(train_samples)
        train_targets = train_samples[:, :, 3:4]
        
        print(f"   Training samples: {train_samples.shape}")
        
        key, tk = jr.split(key)
        trained_ae, hist_ae = klax.fit(
            model.encoder,
            (train_samples, train_targets),
            batch_size=64,
            batch_axis=None,
            steps=STEPS_AE_WARMUP,
            loss_fn=ae_reconstruction_loss,
            optimizer=optax.adam(LR_AE),
            batcher=ae_batcher,
            history=klax.HistoryCallback(log_every=100),
            key=tk
        )
        
        model = eqx.tree_at(
            lambda m: (m.encoder, m.decoder),
            model,
            (trained_ae, trained_ae)
        )
        
        print("AE Warmup complete)")
        

        # PHASE 1: SINGLE-STEP

        print("\n" + "="*80)
        print("PHASE 1: SINGLE-STEP DYNAMICS")
        print("="*80)

        features_t_all = []
        features_next_all = []
        ts_pairs_all = []

        for traj in train_features:
            pairs, ts_t, ts_next = to_step_pairs(traj, ts)
            for i, (feat_t, feat_next) in enumerate(pairs):
                features_t_all.append(feat_t)
                features_next_all.append(feat_next)
                ts_pairs_all.append(jnp.array([ts_t[i], ts_next[i]]))

        features_t_list = jnp.stack(features_t_all)
        features_next_list = jnp.stack(features_next_all)
        ts_pairs = jnp.stack(ts_pairs_all)

        key, dk1 = jr.split(key)
        model, hist_phase1 = klax.fit(
            model,
            (ts_pairs, features_t_list, features_next_list),
            batch_size=BATCH_SIZE,
            batch_axis=None,
            steps=STEPS_PHASE1,
            loss_fn=trajectory_loss_phase1, 
            optimizer=optax.adam(LR_DYN1),
            batcher=step_batcher_phase1,
            history=klax.HistoryCallback(log_every=100),
            key=dk1
        )
 

        # PHASE 2: FULL TRAJECTORY

        print("\n" + "="*80)
        print("PHASE 2: Full TRAJECTORY Training")
        print("="*80)

        key, dk2 = jr.split(key)
        model, hist_phase2 = klax.fit(
            model,
            (train_features, ts),
            batch_size=BATCH_SIZE,
            batch_axis=None,
            steps=STEPS_PHASE2,
            loss_fn=trajectory_loss_phase2,
            optimizer=optax.adam(LR_DYN2),
            batcher=trajectory_batcher_phase2,
            history=klax.HistoryCallback(log_every=100),
            key=dk2
        )

        MODEL_DIR.mkdir(exist_ok=True)
        eqx.tree_serialise_leaves(MODEL_PATH, model)
        
        if hist_phase2:
            hist_phase2.save(HIST_PATH, overwrite=True)
            
        print(f" training complete: {MODEL_PATH}")

    # EVALUATION
    final_model = klax.finalize(model)
    total_time = time.time() - start_total
    
    # Plot training curves
    if hist_phase2 is not None:
        pass
    else:
        plot_training_curves(hist_ae, hist_phase1, hist_phase2)
  
    
    # Test trajectory prediction
    traj_idx = 2
    test_traj = test_features[traj_idx]
    
    # 4 Plots
    plot_1_first_10_latent_variables_training(final_model.encoder, train_features, ts, traj_idx=0)
    plot_2_first_10_field_values_reconstruction(final_model.encoder, train_features, stats, ts, traj_idx=0)
    plot_3_first_10_latent_variables_test_set(final_model, test_features, ts, traj_idx=0)
    plot_4_relaxation_prediction_u0(final_model, test_features, stats, ts, traj_idx=0, start_step=50)
    
    # 3D Mesh Visualization
    traj_idx = 2
    time_idx = 70

    temps_pred = final_model.predict_trajectory(ts, test_features[traj_idx])
    pred_phys = temps_pred[time_idx].flatten() * stats['temp_scale'] + stats['temp_shift']
    true_phys = test_features[traj_idx, time_idx, :, 3] * stats['temp_scale'] + stats['temp_shift']

    mesh_viz = pv.read(DATA_DIR / "mesh.nas")

    mesh_viz["Prediction"] = np.array(pred_phys)
    mesh_viz["True"] = np.array(true_phys)
    mesh_viz["Error"] = np.abs(np.array(pred_phys) - np.array(true_phys))

    p = pv.Plotter(shape=(1, 3), window_size=[1400, 450], notebook=False)

    p.subplot(0, 0)
    p.add_mesh(
        mesh_viz.copy(),
        scalars="Prediction",
        cmap="cps:seismic",
        clim=(mesh_viz["True"].min(), mesh_viz["True"].max()),
    )
    p.add_text(f"MLP Prediction [K]\nt={ts[time_idx]:.3f}s", font_size=10)
    p.view_vector([-1, -1, 1])

    p.subplot(0, 1)
    p.add_mesh(mesh_viz.copy(), scalars="True", cmap="cps:seismic")
    p.add_text("Ground Truth [K]", font_size=10)
    p.view_vector([-1, -1, 1])

    p.subplot(0, 2)
    p.add_mesh(mesh_viz.copy(), scalars="Error", cmap="hot", clim=(0, mesh_viz["Error"].max()))
    p.add_text(f"Absolute Error [K]\nMax: {mesh_viz['Error'].max():.2f}K", font_size=10)
    p.view_vector([-1, -1, 1])

    p.link_views()
    p.show()
    p.close()
    
    print(f"ALL VISUALIZATIONS COMPLETE!")
    print(f"   Total Time: {total_time/60:.1f} min")


if __name__ == "__main__":
    main()
