import cyberplotstyle as cps
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
import jraph
import matplotlib.pyplot as plt
from pathlib import Path
import time
from jax import random as jr
import klax
import jax.tree_util as jtu
import pyvista as pv
from dynax import ISPHS, ConvexLyapunov, ODESolver
from jax.nn.initializers import variance_scaling
from sklearn.cluster import KMeans
from matplotlib.lines import Line2D

# DATA LOADING 

def load_mesh_edges(mesh_file):
    if not mesh_file.exists():
        raise FileNotFoundError(f"Mesh nicht gefunden: {mesh_file}")
    mesh = pv.read(str(mesh_file))
    edges = mesh.extract_all_edges()
    lines = edges.lines.reshape(-1, 3)
    edge_list = lines[:, 1:3]
    return edge_list, mesh.points

def build_sparse_graph_from_edges(nodes, senders, receivers, features, source):
    # Features direkt konkatenieren ohne Python-Listen
    node_features = jnp.concatenate([nodes, features, source], axis=1) 
    # Convert to JAX arrays directly
    graph = jraph.GraphsTuple(
        nodes=jnp.array(node_features, dtype=jnp.float32),
        edges=jnp.ones((len(senders), 1), dtype=jnp.float32),
        senders=jnp.array(senders, dtype=jnp.int32),
        receivers=jnp.array(receivers, dtype=jnp.int32),
        globals=jnp.zeros((1, 1)),
        n_node=jnp.array([N]),
        n_edge=jnp.array([len(senders)])
    )
    return graph

def load_all_ded_trajectories(data_dir, downsample=1, use_mesh_edges=True, trajectories=None):
    raw_data = np.load(data_dir / "data.npz")
    nodes = np.array(raw_data["nodes"])
    temperatures = np.array(raw_data["temp"])
    sources = np.array(raw_data["source"])
    
    n_trajectories, n_timesteps, n_nodes = temperatures.shape
    nodes_orig = n_nodes
    
    if downsample > 1:
        nodes = nodes[::downsample]
        temperatures = temperatures[:, :, ::downsample]

    edges = None
    if use_mesh_edges:
        mesh_file = data_dir / "mesh.nas"
        if mesh_file.exists():
            edges, _ = load_mesh_edges(mesh_file)
            if downsample > 1:
                valid_nodes = set(range(0, nodes_orig, downsample))
                edge_mask = np.isin(edges[:, 0], list(valid_nodes)) & \
                           np.isin(edges[:, 1], list(valid_nodes))
                edges = edges[edge_mask] // downsample

    if trajectories is None:
        trajectories = list(range(n_trajectories))
    
    all_graphs = []
    for traj_idx in trajectories:
        temp_traj = temperatures[traj_idx]
        source_traj = sources[traj_idx]
        traj_graphs = []
        for t in range(n_timesteps):
            temp_t = temp_traj[t].reshape(-1, 1)
            source_t = source_traj[t].reshape(-1, 1)
            if edges is not None:
                g = build_sparse_graph_from_edges(nodes, edges, temp_t, source_t)
            else:
                raise NotImplementedError("k-NN fallback not implemented")
            traj_graphs.append(g)
        all_graphs.append(traj_graphs)

    all_nodes_data = np.concatenate([np.array(g.nodes) for traj in all_graphs for g in traj], axis=0)
    
    # Position normalization
    pos_data = all_nodes_data[:, :3]
    pos_mean = pos_data.mean(axis=0)
    pos_std = np.where(pos_data.std(axis=0) > 1e-8, pos_data.std(axis=0), 1.0)
    
    # TEMPERATURE: (shift=293, scale=std)
    temp_data = all_nodes_data[:, 3:4]
    temp_shift = 293.0  # Ambient temperature → 0 
    temp_std = float(temp_data.std())
    
    # Source normalization
    source_data = all_nodes_data[:, 4:5]
    source_max = float(source_data.max())
    if source_max < 1e-8: source_max = 1.0

    stats = {
        'pos_mean': jnp.array(pos_mean),
        'pos_std': jnp.array(pos_std),
        'temp_shift': jnp.array(temp_shift),  
        'temp_scale': jnp.array(temp_std),
        'source_max': jnp.array(source_max),
        'temp_min': float(temp_data.min()),
        'temp_max': float(temp_data.max())
    }

    all_graphs_norm = []
    for traj_graphs in all_graphs:
        traj_norm = []
        for g in traj_graphs:
            pos_norm = (g.nodes[:, :3] - stats['pos_mean']) / stats['pos_std']
            temp_norm = (g.nodes[:, 3:4] - stats['temp_shift']) / stats['temp_scale'] 
            source_norm = g.nodes[:, 4:5] / stats['source_max']
            
            nodes_norm = jnp.concatenate([pos_norm, temp_norm, source_norm], axis=1)
            traj_norm.append(g._replace(nodes=nodes_norm))
        all_graphs_norm.append(traj_norm)
        
    metadata = {
        'n_trajectories': len(trajectories),
        'n_timesteps': n_timesteps,
        'n_nodes': nodes.shape[0],
        'trajectory_indices': trajectories,
        'feature_names': ['x', 'y', 'z', 'temp', 'source']
    }
    
    return all_graphs_norm, stats, all_graphs, metadata


def create_geometric_assignment(graphs, n_clusters):
    positions = np.array(graphs[0].nodes[:, :3]) 
    N = positions.shape[0]
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(positions)
    
    S = np.zeros((N, n_clusters), dtype=np.float32)
    for i in range(N):
        S[i, labels[i]] = 1.0
    
    return jnp.array(S)

# GAE LAYERS

class DenseGCNLayer(eqx.Module):
    linear: eqx.nn.Linear
    use_activation: bool = eqx.field(static=True)

    def __init__(self, in_features, out_features, key, use_activation=True):
        self.linear = eqx.nn.Linear(in_features, out_features, key=key)
        self.use_activation = use_activation

    def __call__(self, x, adj):
        adj = adj + jnp.eye(adj.shape[0])
        degree = jnp.sum(adj, axis=1)
        degree = jnp.maximum(degree, 1.0)
        deg_inv_sqrt = jnp.power(degree, -0.5)
        adj_norm = adj * deg_inv_sqrt[:, None] * deg_inv_sqrt[None, :]
        
        h = jnp.matmul(adj_norm, x)
        h = jax.vmap(self.linear)(h)
        
        return jax.nn.leaky_relu(h) if self.use_activation else h


class GraphConvLayer(eqx.Module):
    linear: eqx.nn.Linear
    use_activation: bool = eqx.field(static=True)

    def __init__(self, in_features, out_features, key, use_activation=True):
        self.linear = eqx.nn.Linear(in_features, out_features, key=key)
        self.use_activation = use_activation

    def __call__(self, graph):
        def update_node_fn(aggregated):
            return jax.vmap(self.linear)(aggregated)

        g = jraph.GraphConvolution(
            update_node_fn=update_node_fn,
            aggregate_nodes_fn=jraph.segment_sum,
            add_self_edges=True,
            symmetric_normalization=True
        )(graph)

        nodes = jax.nn.leaky_relu(g.nodes) if self.use_activation else g.nodes
        return g._replace(nodes=nodes)


class StructPool(eqx.Module):
    S: jnp.ndarray
    k: int = eqx.field(static=True)

    def __init__(self, S_matrix):
        self.S = S_matrix
        self.k = S_matrix.shape[1]

    def pool_nodes(self, nodes):
        cluster_sizes = jnp.sum(self.S, axis=0)
        safe_sizes = jnp.where(cluster_sizes > 0, cluster_sizes, 1.0)
        S_norm = self.S / safe_sizes[None, :]
        return jnp.matmul(S_norm.T, nodes)


class DiffUnpool(eqx.Module):
    def __call__(self, nodes_pooled, S):
        return jnp.matmul(S, nodes_pooled)


# GAE AUTOENCODER
class GAEAutoEncoder(eqx.Module):
    static_A1: jnp.ndarray
    static_A2: jnp.ndarray 
    
    enc_gcn_sparse: GraphConvLayer   
    pool1: StructPool                
    enc_gcn_dense1: DenseGCNLayer    
    pool2: StructPool                
    enc_gcn_dense2: DenseGCNLayer    
    enc_mlp_to_latent: eqx.nn.MLP    # OHNE LayerNorm!
    
    dec_mlp_from_latent: eqx.nn.MLP
    dec_gcn_dense2: DenseGCNLayer    
    unpool2: DiffUnpool              
    dec_gcn_dense1: DenseGCNLayer    
    unpool1: DiffUnpool              
    dec_gcn_sparse: GraphConvLayer   
    
    latent_dim: int = eqx.field(static=True)
    pool2_nodes: int = eqx.field(static=True)
    pool2_features: int = eqx.field(static=True)
    n_nodes: int = eqx.field(static=True)

    def __init__(self, in_features, latent_dim, S1, S2, senders, receivers, n_nodes, key):
        keys = jax.random.split(key, 10)
        self.latent_dim = latent_dim
        self.pool2_nodes = S2.shape[1]   
        self.pool2_features = 64
        self.n_nodes = n_nodes
        
        adj = jnp.zeros((n_nodes, n_nodes)).at[senders, receivers].set(1.0)
        self.static_A1 = jnp.matmul(jnp.matmul(S1.T, adj), S1) + jnp.eye(S1.shape[1])
        self.static_A2 = jnp.matmul(jnp.matmul(S2.T, self.static_A1), S2) + jnp.eye(S2.shape[1])
        
        self.enc_gcn_sparse = GraphConvLayer(in_features, 16, keys[0], True)
        self.pool1 = StructPool(S1)
        self.enc_gcn_dense1 = DenseGCNLayer(16, 32, keys[1], True)
        self.pool2 = StructPool(S2)
        self.enc_gcn_dense2 = DenseGCNLayer(32, 64, keys[2], True)
        
        self.enc_mlp_to_latent = eqx.nn.MLP(self.pool2_nodes * 64, latent_dim, 256, 2, key=keys[3])
        
        self.dec_mlp_from_latent = eqx.nn.MLP(latent_dim, self.pool2_nodes * 64, 256, 2, key=keys[4])
        self.dec_gcn_dense2 = DenseGCNLayer(64, 32, keys[5], True)
        self.unpool2 = DiffUnpool()
        self.dec_gcn_dense1 = DenseGCNLayer(32, 16, keys[6], True)
        self.unpool1 = DiffUnpool()
        self.dec_gcn_sparse = GraphConvLayer(16, 1, keys[7], False)

    def encode(self, graph):
        """Encode WITHOUT LayerNorm - keeps physical meaning!"""
        x = self.enc_gcn_sparse(graph).nodes
        x = self.pool1.pool_nodes(x)
        x = self.enc_gcn_dense1(x, self.static_A1)
        x = self.pool2.pool_nodes(x)
        x = self.enc_gcn_dense2(x, self.static_A2)
        
        z = self.enc_mlp_to_latent(x.reshape(-1))  # Direct projection, NO LayerNorm!
        return z
    
    def decode(self, z):
        x = self.dec_mlp_from_latent(z).reshape(self.pool2_nodes, 64)
        x = self.dec_gcn_dense2(x, self.static_A2)
        x = self.unpool2(x, self.pool2.S)
        x = self.dec_gcn_dense1(x, self.static_A1)
        x = self.unpool1(x, self.pool1.S)
        return x

    def __call__(self, graph):
        z = self.encode(graph)
        x = self.decode(z)
        g_final = graph._replace(nodes=x)
        return self.dec_gcn_sparse(g_final).nodes

# LASER ENCODER
class LaserInputEncoder(eqx.Module):
    mlp: eqx.nn.MLP
    output_dim: int = eqx.field(static=True)
    
    def __init__(self, key, output_dim=8):
        self.output_dim = output_dim
        self.mlp = eqx.nn.MLP(4, output_dim, 16, 2, activation=jax.nn.leaky_relu, key=key)
    
    def __call__(self, nodes_with_source):
        positions = nodes_with_source[:, :3]
        source = nodes_with_source[:, 4:5]
        
        max_idx = jnp.argmax(source)
        laser_pos = positions[max_idx]
        laser_intensity = source[max_idx, 0]
        
        laser_features = jnp.concatenate([laser_pos, jnp.array([laser_intensity])])
        u = self.mlp(laser_features)
        return u

# sPHNN DYNAMICS

class GAEsPHNNFullModel(eqx.Module):
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
            epsilon=0.0, 
            init=variance_scaling(1, "fan_avg", "truncated_normal"),
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

    def __call__(self, ts, z0, all_nodes_sequence):
        us = jax.vmap(self.laser_encoder)(all_nodes_sequence)
        return self.solver(ts, z0, us)

# COMBINED MODEL

class GAEsPHNNFullModel(eqx.Module):
    encoder: GAEAutoEncoder
    decoder: GAEAutoEncoder
    dynamics: GAEsPHNNFullModel
    
    def __init__(self, key, ae_architecture: GAEAutoEncoder, latent_dim=32):
        self.encoder = ae_architecture
        self.decoder = ae_architecture
        self.dynamics = GAEsPHNNFullModel(key, state_size=latent_dim, laser_latent_size=8)
    
    def predict_trajectory(self, ts, initial_graph, graphs_sequence):
        z0 = self.encoder.encode(initial_graph)
        nodes_seq = jnp.stack([g.nodes for g in graphs_sequence])
        zs = self.dynamics(ts, z0, nodes_seq)
        
        def decode_single(z, graph_template):
            x = self.decoder.decode(z)
            g_temp = graph_template._replace(nodes=x)
            return self.decoder.dec_gcn_sparse(g_temp).nodes
        
        return jax.vmap(decode_single, in_axes=(0, None))(zs, initial_graph)


# LOSS

def trajectory_loss(model, data, batch_axis):

    if len(data) == 3:
        # PHASE 1: Single-step
        ts_pairs, graphs_t, graphs_next = data
        
        if isinstance(graphs_t, list):
            nodes_t = jnp.stack([g.nodes for g in graphs_t])
            nodes_next = jnp.stack([g.nodes for g in graphs_next])
            template_graph = graphs_t[0]
        else:
            nodes_t = graphs_t.nodes
            nodes_next = graphs_next.nodes
            template_graph = jtu.tree_map(lambda x: x[0], graphs_t)
        
        def single_step_loss(t_pair, n_t, n_next):
            g_t = template_graph._replace(nodes=n_t)
            g_next = template_graph._replace(nodes=n_next)
            
            z_t = jax.lax.stop_gradient(model.encoder.encode(g_t))
            z_true_next = jax.lax.stop_gradient(model.encoder.encode(g_next))
            
            ts_local = jnp.array([0.0, t_pair])
            nodes_seq = jnp.stack([n_t, n_next])
            z_pred = model.dynamics(ts_local, z_t, nodes_seq)[-1]
            
            return jnp.mean((z_pred - z_true_next)**2)
        
        dt = ts_pairs[:, 1] - ts_pairs[:, 0]
        losses = jax.vmap(single_step_loss)(dt, nodes_t, nodes_next)
        return jnp.mean(losses)
    
    else:
        # PHASE 2: Full trajectory
        trajectory_list, ts_traj = data
        
        if isinstance(trajectory_list, list):
            batch_nodes = jnp.stack([jnp.stack([g.nodes for g in traj]) for traj in trajectory_list])
            template = trajectory_list[0][0]
        else:
            batch_nodes = trajectory_list
            template = jtu.tree_map(lambda x: x[0, 0], trajectory_list)
        
        def single_traj_loss(nodes_traj):
            def encode_frame(nodes):
                return jax.lax.stop_gradient(model.encoder.encode(template._replace(nodes=nodes)))
            
            z_true = jax.vmap(encode_frame)(nodes_traj)
            
            z0 = z_true[0]
            z_pred = model.dynamics(ts_traj, z0, nodes_traj)

            return jnp.mean((z_pred - z_true)**2)
        
        return jnp.mean(jax.vmap(single_traj_loss)(batch_nodes))


# AE WARMUP LOSS - Only for pre-training encoder

def ae_reconstruction_loss(model, data, batch_axis):
    graphs, _ = data
    
    if isinstance(graphs, list):
        graphs_batch = jtu.tree_map(lambda *xs: jnp.stack(xs), *graphs)
    else:
        graphs_batch = graphs
    
    def single_recon_loss(g):
        pred = model(g)
        true = g.nodes[:, 3:4]
        return jnp.mean((pred - true)**2)
    
    if isinstance(graphs_batch.nodes, jnp.ndarray) and graphs_batch.nodes.ndim > 2:
        return jnp.mean(jax.vmap(single_recon_loss, in_axes=0)(graphs_batch))
    else:
        return single_recon_loss(graphs_batch)

# BATCHER

def graph_batcher(data, batch_size=64, batch_axis=0, convert_to_numpy=True, *, key):
    graphs, targets = data
    n_samples = len(graphs)
    indices = jnp.arange(n_samples)

    while True:
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, indices)

        for i in range(0, n_samples, batch_size):
            batch_idx = perm[i : i + batch_size]
            batch_graphs_list = [graphs[int(idx)] for idx in batch_idx]
            batch_targets_list = [targets[int(idx)] for idx in batch_idx]
            
            batch_graphs = jtu.tree_map(lambda *xs: jnp.stack(xs), *batch_graphs_list)
            batch_targets = jnp.stack(batch_targets_list)

            yield (batch_graphs, batch_targets)


def step_batcher_phase1(data, batch_size=8, batch_axis=None, convert_to_numpy=True, *, key):
    ts_pairs, graphs_t_list, graphs_next_list = data
    n_samples = len(graphs_t_list)
    
    while True:
        key, subkey = jax.random.split(key)
        indices = jax.random.permutation(subkey, n_samples)
        
        for i in range(0, n_samples - batch_size + 1, batch_size):
            idx = indices[i : i + batch_size]
            
            batch_ts = jnp.stack([ts_pairs[int(j)] for j in idx])
            batch_gt = jtu.tree_map(lambda *xs: jnp.stack(xs), 
                                    *[graphs_t_list[int(j)] for j in idx])
            batch_gn = jtu.tree_map(lambda *xs: jnp.stack(xs), 
                                    *[graphs_next_list[int(j)] for j in idx])
            
            yield (batch_ts, batch_gt, batch_gn)


def trajectory_batcher_phase2(data, batch_size=8, batch_axis=None, convert_to_numpy=True, *, key):
    trajectory_list, ts_traj = data
    n_trajectories = len(trajectory_list)
    
    while True:
        key, subkey = jax.random.split(key)
        indices = jax.random.permutation(subkey, n_trajectories)
        
        for i in range(0, n_trajectories - batch_size + 1, batch_size):
            idx = indices[i : i + batch_size]
            
            batch_trajs = [trajectory_list[int(j)] for j in idx]
            
            yield (batch_trajs, ts_traj)


def to_step_pairs(graphs_list, ts):
    n_graphs = len(graphs_list)
    pairs = []
    for i in range(n_graphs - 1):
        pairs.append((graphs_list[i], graphs_list[i+1]))
    return pairs, ts[:-1], ts[1:]


# MAIN

def main():
   
    SEED = 42
    LATENT = 30 
    STEPS_AE_WARMUP = 5000  
    STEPS_PHASE1 = 10000    
    STEPS_PHASE2 = 4000      
    
    BATCH_SIZE = 8
    LR_AE = 5e-4
    LR_DYN = 6e-4
    
    N_TRAJECTORIES_TO_USE = 25
    MODEL_DIR = Path("gae_models")
    DATA_DIR = Path("ded_data")
    
    start_total = time.time()


    all_graphs_norm, stats, all_graphs_raw, metadata = load_all_ded_trajectories(
        DATA_DIR, downsample=1, use_mesh_edges=True,
        trajectories=list(range(N_TRAJECTORIES_TO_USE))
    )
    
    n_nodes = metadata['n_nodes']
    raw_data = np.load(DATA_DIR / "data.npz")
    ts = jnp.array(raw_data["ts"])


    train_graphs = []
    for traj in all_graphs_norm[:20]:
        train_graphs.extend(traj)

    k1, k2 = 400, 64
        
    S1 = create_geometric_assignment(train_graphs, n_clusters=k1)
    
    positions_orig = np.array(train_graphs[0].nodes[:, :3])
    cluster_sizes_1 = np.sum(S1, axis=0)
    S1_norm = S1 / np.where(cluster_sizes_1 > 0, cluster_sizes_1, 1.0)[None, :]
    positions_pooled_1 = S1_norm.T @ positions_orig
    
    kmeans_2 = KMeans(n_clusters=k2, random_state=SEED, n_init=10)
    labels_2 = kmeans_2.fit_predict(positions_pooled_1)
    S2 = np.zeros((k1, k2), dtype=np.float32)
    for i in range(k1): 
        S2[i, labels_2[i]] = 1.0
    S2 = jnp.array(S2)

    MODEL_PATH = MODEL_DIR / "sPHNN_GAE.eqx"
    HIST_PATH = MODEL_DIR / "sPHNN_GAE_hist.pkl"

    key = jr.PRNGKey(SEED)
    key, mk, dk = jr.split(key, 3)
    
    ref_g = all_graphs_norm[0][0]
    senders, receivers = ref_g.senders, ref_g.receivers

    ae_architecture = GAEAutoEncoder(
        in_features=5, latent_dim=LATENT, S1=S1, S2=S2,
        senders=senders, receivers=receivers, n_nodes=int(n_nodes), key=mk
    )
    
    dyn_model = GAEsPHNNFullModel(dk, ae_architecture, latent_dim=LATENT)

    if MODEL_PATH.exists():
        print(f"\n LOADING TRAINED MODEL: {MODEL_PATH}")
        dyn_model = eqx.tree_deserialise_leaves(MODEL_PATH, dyn_model)
        hist_phase2 = klax.HistoryCallback.load(HIST_PATH)
    else:
        print("\n NO MODEL FOUND: Starting training...")
        
        # PHASE 0: AE WARMUP
        
        train_targets = [g.nodes[:, 3:4] for g in train_graphs]
        train_data = (train_graphs, train_targets)
        
        key, tk = jr.split(key)
        trained_ae, hist_ae = klax.fit(
            dyn_model.encoder, 
            train_data,
            batch_size=64,
            batch_axis=None,
            steps=STEPS_AE_WARMUP,
            loss_fn=ae_reconstruction_loss,
            optimizer=optax.adam(LR_AE),
            batcher=graph_batcher,
            history=klax.HistoryCallback(log_every=100),
            key=tk
        )
        
        dyn_model = eqx.tree_at(
            lambda m: (m.encoder, m.decoder),
            dyn_model,
            (trained_ae, trained_ae)
        )

        # PHASE 1: SINGLE-STEP (Encoder FROZEN)
        
        all_step_pairs = []
        all_ts_pairs = []
        for traj in all_graphs_norm[:20]:
            pairs, ts_t, ts_next = to_step_pairs(traj, ts)
            for i, (g_t, g_next) in enumerate(pairs):
                all_step_pairs.append((g_t, g_next))
                all_ts_pairs.append(jnp.array([ts_t[i], ts_next[i]]))
        
        graphs_t_list = [p[0] for p in all_step_pairs]
        graphs_next_list = [p[1] for p in all_step_pairs]
        ts_pairs = jnp.stack(all_ts_pairs)
        
        key, dk1 = jr.split(key)
        dyn_model, hist_phase1 = klax.fit(
            dyn_model,
            (ts_pairs, graphs_t_list, graphs_next_list),
            batch_size=BATCH_SIZE,
            batch_axis=None,
            steps=STEPS_PHASE1,
            loss_fn=trajectory_loss,
            optimizer=optax.adam(LR_DYN),
            batcher=step_batcher_phase1,
            history=klax.HistoryCallback(log_every=100),
            key=dk1
        )

        # PHASE 2: FULL TRAJECTORY (Encoder FROZEN)

        train_sequences = all_graphs_norm[:20]

        key, dk2 = jr.split(key)
        dyn_model, hist_phase2 = klax.fit(
            dyn_model,
            (train_sequences, ts),
            batch_size=BATCH_SIZE,
            batch_axis=None,
            steps=STEPS_PHASE2,
            loss_fn=trajectory_loss, 
            optimizer=optax.adam(LR_DYN),
            batcher=trajectory_batcher_phase2,
            history=klax.HistoryCallback(log_every=100),
            key=dk2
        )

        MODEL_DIR.mkdir(exist_ok=True)
        eqx.tree_serialise_leaves(MODEL_PATH, dyn_model)
        
        if hist_phase2:
            hist_phase2.save(HIST_PATH, overwrite=True)

    
    final_model = klax.finalize(dyn_model)
    total_time = time.time() - start_total
    print(f"\n {total_time/60:.1f} min")
    total_time = time.time() - start_total

if __name__ == "__main__":
    main()
