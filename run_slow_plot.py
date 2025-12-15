import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# ==============================
# ID <-> (i,j) mapping (y,x)
# ==============================
def ij_to_id(i: int, j: int, side: int) -> int:
    """Deterministic mapping: tree_id = j + side * i."""
    return int(j) + int(side) * int(i)

def id_to_ij(tree_id: int, side: int) -> tuple[int, int]:
    """Inverse mapping: (i,j) from tree_id."""
    i = int(tree_id) // int(side)
    j = int(tree_id) % int(side)
    return i, j

# ==============================
# Helpers
# ==============================
def add_tree_at(network, grid, i: int, j: int, init_biomass=1.0):
    side = grid["N"]
    if grid["biomass"][i, j] > 0:
        return None  # already has a tree

    tree_id = ij_to_id(i, j, side)

    # grid
    grid["biomass"][i, j] = float(init_biomass)
    grid["carbon_intake"][i, j] = 0.0

    # network
    network.add_node(tree_id)
    return tree_id


def remove_tree_at(network, grid, i: int, j: int):
    """
    Remove the tree at grid cell (i,j) from both grid and network.
    """
    side = grid["N"]

    if not (0 <= i < side and 0 <= j < side):
        return False  # out of bounds

    if grid["biomass"][i, j] <= 0:
        # already empty (or already "dead" on grid)
        # still ensure network doesn't have the node
        tree_id = ij_to_id(i, j, side)
        if network.has_node(tree_id):
            network.remove_node(tree_id)
        return False
    
    tree_id = ij_to_id(i, j, side)

    # Clear grid
    grid["biomass"][i, j] = 0.0
    grid["carbon_intake"][i, j] = 0.0

    # clear network
    if network.has_node(tree_id):
        network.remove_node(tree_id)

    return True


def grow_seedlings(prob_seedling, network, grid, init_biomass=1.0):
    """
    Grow seedlings on the grid with probability, and add the new seedlings to the network layer
    """
    N = grid["N"]
    biomass = grid["biomass"]

    rng = grid["rng"]

    # empty cells mask
    empty = biomass <= 0.0
    if not np.any(empty):
        return 0

    # sample new seedlings
    born = empty & (rng.random((N, N)) < float(prob_seedling))
    if not np.any(born):
        return 0

    # add them
    new_count = 0
    for i, j in np.argwhere(born):
        tid = add_tree_at(network, grid, int(i), int(j), init_biomass=float(init_biomass))
        if tid is not None:
            new_count += 1

    return new_count

def add_connections(network, grid, scale_free_alpha, m=2, delta=1.0, beta=1.0):
    """
    Add new mycorrizal network connections on the network layer using scale-free network building method
    """
    forest = grid["biomass"]
    rng = grid["rng"]

    # Treat nodes with degree 0 as "new seedlings" to be attached this step
    new_nodes = [n for n, deg in network.degree() if deg == 0]

    # If you start from an empty/isolated state, nothing to attach to
    if len(network) <= 1:
        return

    for v in new_nodes:
        preferential_attachment(
            network=network,
            forest=forest,
            new_node=v,
            scale_free_alpha=scale_free_alpha,
            rng=rng,
            m=m,
            delta=delta,
            beta=beta,
        )


def periodic_distance(p1, p2, shape):
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    shape = np.asarray(shape)

    delta = np.abs(p1 - p2)

    delta = np.minimum(delta, shape - delta)

    return np.sqrt(np.sum(delta**2))



def preferential_attachment(network, forest, new_node, scale_free_alpha, rng, m, delta, beta):
    """
    Attach 'new_node' to m existing nodes.
    P(v->i): (k_i + delta)^beta * exp(-alpha * dist / (biomass_i + eps))
    """
    side_len = len(forest)
    p1 = id_to_ij(new_node, side_len)

    # candidates: all other nodes except itself
    candidates = [n for n in network.nodes if n != new_node]
    if not candidates:
        return network

    # number of edges to add (can't exceed number of candidates)
    m_eff = min(int(m), len(candidates))
    if m_eff <= 0:
        return network
    

    eps = 1e-12
    chosen = set()

    for _ in range(m_eff):
        # build weights for remaining candidates
        remaining = [n for n in candidates if n not in chosen]
        if not remaining:
            break

        weights = []
        for n in remaining:
            p2 = id_to_ij(n, side_len)
            i2, j2 = p2[0], p2[1]

            dist = periodic_distance(p1, p2, np.shape(forest))
            biomass_i = float(forest[i2, j2])

            k = network.degree(n)
            pref = (k + float(delta)) ** float(beta)   # degree-driven rich-get-richer
            eco  = np.exp(- float(scale_free_alpha) * dist / (biomass_i + eps))  # distance+biomass

            w = pref * eco
            weights.append(w)

        weights = np.asarray(weights, dtype=float)

        # fallback if all weights are zero/NaN (rare but can happen)
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            idx = int(rng.integers(0, len(remaining)))
        else:
            probs = weights / weights.sum()
            cdf = np.cumsum(probs)
            idx = int(np.searchsorted(cdf, rng.random(), side="right"))
            if idx >= len(remaining):
                idx = len(remaining) - 1

        target = remaining[idx]
        network.add_edge(new_node, target)
        chosen.add(target)
    
    return network

def calculate_C_intake(grid, env_stress, c_rate: float = 1.0):
    """
    C intake in each step: carbon_intake = c_rate * biomass / (1 + env_stress)
    Only trees (biomass > 0) produce intake; empty cells stay 0

    Returns:
      Updated grid["carbon_intake"]
    """
    biomass = grid["biomass"]
    carbon_intake = grid["carbon_intake"]
    max_carbon_intake = carbon_intake.copy()

    denom = 1.0 + max(float(env_stress), 0.0)
    carbon_intake.fill(0.0)   # Clear memory each step
    mask = biomass > 0.0
    max_carbon_intake[mask] = float(c_rate) * biomass[mask]
    carbon_intake[mask] = float(c_rate) * biomass[mask] / denom

    return carbon_intake, max_carbon_intake


def allocate_C_intake(network, grid, C_intake_grid, max_carbon_intake_grid):
    """
    Use diffusion model to allocate C
    Return a map grid with final biomass growth
    """

    #get subgraphs
    subgraphs = list(nx.connected_components(network))
    N = grid["N"]
    before_allocation = C_intake_grid.copy()

    for s in subgraphs:
        carbon_sum = 0
        cmax_value = []
        for n in s:
            i,j = id_to_ij(n, N)
            carbon_sum += float(C_intake_grid[i,j])
            cmax_value.append(max_carbon_intake_grid[i,j])
        
        pairs = sorted(zip(s, cmax_value), key=lambda x: x[1])
        s_ordered, cmax_ordered = (zip(*pairs) if pairs else ((), ()))
        # s_ordered, cmax_ordered = zip(*sorted(zip(s,cmax_value)))

        for n in range(len(s_ordered)):
            avg_C = carbon_sum / (len(s_ordered) - n)
            i,j = id_to_ij(s_ordered[n], N)
            if(cmax_ordered[n] < avg_C):
                C_intake_grid[i,j] = cmax_ordered[n]
                carbon_sum -= cmax_ordered[n]
            else:
                C_intake_grid[i,j] = avg_C
                carbon_sum -= avg_C

    return before_allocation, C_intake_grid
                



def check_survival(network, grid, env_stress):
    """
    1) Remove any trees with biomass <= 0 (both grid and network)
    2) Random death with probability: death_prob = clip(env_stress, 0, 1)
       For each remaining tree, kill with death_prob.
    
    Also sync surviving nodes' attributes (biomass, carbon_intake) from grid.

    Returns:
      removed_count: int
    """
    N = grid["N"]
    biomass = grid["biomass"]
    carbon = grid["carbon_intake"]

    rng = grid["rng"]

    removed = 0

    # step 1)
    dead_cells = np.argwhere(biomass <= 0.0)
    for i, j in dead_cells:
        # only count if it actually existed in either layer
        tid = ij_to_id(int(i), int(j), N)
        existed = (biomass[int(i), int(j)] > 0.0) or network.has_node(tid)
        ok = remove_tree_at(network, grid, int(i), int(j))
        if ok or existed:
            removed += 1

    # step 2) stochastic removals by env_stress
    death_prob = 0.05 * env_stress             # ------------------------- modified!
    death_prob = np.clip(death_prob, 0, 1)     # ------------------------- modified!
    if death_prob > 0.0:
        alive_mask = biomass > 0.0
        if np.any(alive_mask):
            die = alive_mask & (rng.random((N, N)) < death_prob)
            for i, j in np.argwhere(die):
                if remove_tree_at(network, grid, int(i), int(j)):
                    removed += 1

    # Keep network consistent with grid after any other operations
    # Grid is the single source of truth for biomass and carbon
    for node in list(network.nodes):
        i, j = id_to_ij(int(node), N)
        if biomass[i, j] <= 0.0:
            network.remove_node(node)

    return removed




def run_simulation(prob_seedling, scale_free_alpha, env_stress, N, steps, seed=None, init_tree_density=0.0, init_biomass=1.0):
    """
    N: grid shape N*N
    seed: random seed (optional)
    """

    # RNG
    rng = np.random.default_rng(seed)

    # Grid layer initialization
    # biomass / carbon_intake: in same shapes
    biomass = np.zeros((N, N), dtype=np.float64)        # 0 => empty
    carbon_intake = np.zeros((N, N), dtype=np.float64)  # workspace


    grid = {
        "N": N,
        "biomass": biomass,
        "carbon_intake": carbon_intake,
        "step": 0,
        "rng": rng, 
    }

    # -----------------------
    # Network layer initialization
    # -----------------------
    network = nx.Graph()


    # optional init
    if init_tree_density > 0:
        mask = rng.random((N, N)) < init_tree_density
        for (i, j) in np.argwhere(mask):
            add_tree_at(network, grid, int(i), int(j), init_biomass)

    # keep snapshot of the initial biomass configuration for plotting/reporting
    grid["initial_biomass"] = grid["biomass"].copy()

    history = {
        "biomass_start": [],     # list of floats
        "carbon_transfer": [],   # list of floats
        "step": [],              # 可选：记录来自哪一步
    }

    # -----------------------
    # Main loop
    # -----------------------
    for t in range(steps):
        grid["step"] = t
        grow_seedlings(prob_seedling, network, grid, init_biomass)
        add_connections(network, grid, scale_free_alpha)

        # ===== 记录“本步开始(增长前)”的生物量快照 =====
        biomass_start = grid["biomass"].copy()

        carbon_intake, max_carbon_intake = calculate_C_intake(grid, env_stress)
        C_grid_former, C_grid_later = allocate_C_intake(network, grid, carbon_intake, max_carbon_intake)

        transfer_grid = C_grid_later - C_grid_former
        alive = biomass_start > 0.0
        if np.any(alive):
            history["biomass_start"].extend(biomass_start[alive].ravel().tolist())
            history["carbon_transfer"].extend(transfer_grid[alive].ravel().tolist())
            history["step"].extend([t] * int(np.sum(alive)))

        grid["biomass"] += C_grid_later
        check_survival(network, grid, env_stress)
        print(t)

    return network, grid, history


def plot_biomass_states(initial_biomass, final_biomass, title="Biomass States"):
    """
    Render side-by-side grid plots showing the biomass distribution at the start and end.
    """
    vmax = max(float(np.max(initial_biomass)), float(np.max(final_biomass)), 1e-12)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    im0 = axes[0].imshow(initial_biomass, cmap="YlGn", vmin=0, vmax=vmax)
    axes[0].set_title("Initial State")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    im1 = axes[1].imshow(final_biomass, cmap="YlGn", vmin=0, vmax=vmax)
    axes[1].set_title("Final State")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    fig.suptitle(title)
    fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.8, label="Biomass")
    fig.tight_layout()
    plt.show()


def plot_transfer_vs_biomass(history, n_bins=8, binning="quantile", title="Carbon transfer vs Biomass"):
    """
    把 history 里的样本按生物量分箱，计算每箱 carbon_transfer 平均值并绘图。
    binning:
      - "quantile": 分位数分箱（推荐，箱内样本更均衡）
      - "linear": 线性等宽分箱
    """
    b = np.asarray(history["biomass_start"], dtype=float)
    tr = np.asarray(history["carbon_transfer"], dtype=float)

    mask = np.isfinite(b) & np.isfinite(tr)
    b = b[mask]
    tr = tr[mask]

    if b.size == 0:
        print("No samples to plot.")
        return

    # 构造 bins
    if binning == "quantile":
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.quantile(b, qs)
        # 防止重复边界导致空箱：做一点去重处理
        edges = np.unique(edges)
        if len(edges) < 3:
            # 生物量几乎都一样，退化成线性分箱
            edges = np.linspace(b.min(), b.max() + 1e-12, max(3, n_bins + 1))
    else:
        edges = np.linspace(b.min(), b.max() + 1e-12, n_bins + 1)

    # 分箱并统计每箱平均 transfer
    idx = np.digitize(b, edges, right=False) - 1
    n_bins_eff = len(edges) - 1

    x_center = []
    y_mean = []
    y_sem = []   # 可选：标准误，能看出噪声大小
    counts = []

    for k in range(n_bins_eff):
        sel = (idx == k)
        if not np.any(sel):
            continue
        bk = b[sel]
        trk = tr[sel]
        x_center.append(0.5 * (edges[k] + edges[k+1]))
        y_mean.append(trk.mean())
        # 标准误（SEM）
        y_sem.append(trk.std(ddof=1) / np.sqrt(trk.size) if trk.size > 1 else 0.0)
        counts.append(trk.size)

    x_center = np.asarray(x_center)
    y_mean = np.asarray(y_mean)
    y_sem = np.asarray(y_sem)

    plt.figure(figsize=(7, 4))
    plt.errorbar(x_center, y_mean, yerr=y_sem, fmt="o-", capsize=3)
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Biomass (binned)")
    plt.ylabel("Mean carbon transfer (C_later - C_former)")
    plt.title(title)
    plt.tight_layout()
    plt.show()

    # 顺手把每箱样本数打印出来，方便你判断分箱是否合理
    print("Bin counts:", counts)


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    network, grid, history = run_simulation(0.4, 1, 0.4, 30, 100, 2, 0.1)
    plot_biomass_states(grid["initial_biomass"], grid["biomass"])
    plot_transfer_vs_biomass(history, n_bins=4, binning="linear", title="Mean carbon transfer vs biomass (quantile bins)")