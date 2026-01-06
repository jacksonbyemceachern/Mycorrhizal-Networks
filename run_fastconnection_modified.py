import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy

# ==============================
# ID <-> (i,j) mapping (y,x)
# ==============================
def ij_to_id(i: int, j: int, side: int) -> int:
    """Deterministic mapping: tree_id = j + side * i."""
    return int(j) + int(side) * int(i)

def id_to_ij(node_id, N):
    """Convert linear node ID to (i,j) in NxN forest array."""
    return divmod(node_id, N)


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


def add_connections(network, grid, scale_free_alpha, m=2, delta=1.0, beta=1.0, eps=1e-12):
    """
    Vectorized and faster preferential attachment with degree + distance.
    Approximate Euclidean distances; optionally periodic boundary can be added.
    """
    forest = grid["biomass"]
    rng = grid["rng"]
    N = forest.shape[0]

    # Get new nodes (degree 0)
    new_nodes = np.fromiter((n for n, deg in network.degree() if deg == 0), dtype=int)
    
    if len(network) <= 1 or new_nodes.size == 0:
        return

    # Precompute positions for all nodes in network
    all_nodes = np.array(list(network.nodes()), dtype=int)                  # (M,)
    coords = np.array([divmod(n, N) for n in all_nodes], dtype=int)         # (M,2) -> (i,j)

    # Degrees as array
    degrees = np.array([network.degree(n) for n in all_nodes], dtype=float) # (M,)

    shape = np.array(forest.shape, dtype=int)  # (2,)

    for v in new_nodes:
        m_eff = min(m, len(all_nodes) - 1)
        if m_eff <= 0:
            continue

        # Position of new node
        p1 = np.array(divmod(v, N), dtype=int)

        # Candidate mask (exclude self)
        mask = all_nodes != v
        candidates = all_nodes[mask]
        candidate_coords = coords[mask]
        candidate_degrees = degrees[mask]

        if candidates.size == 0:
            continue

        # ---- Euclidean distances, periodic boundary!!! ----
        # raw diff
        diff = np.abs(candidate_coords - p1)            # (C,2)
        # wrap-around: min(diff, shape-diff)
        diff = np.minimum(diff, shape - diff)      # (C,2)
        dists = np.linalg.norm(diff.astype(float), axis=1)  # (C,)

        # biomass_i
        bio = forest[candidate_coords[:, 0], candidate_coords[:, 1]].astype(float)    # (C,)

        # Compute attachment weights
        pref = (candidate_degrees + delta) ** beta
        eco  = np.exp(-float(scale_free_alpha) * dists / (bio + eps))
        weights = pref * eco

        # 清理异常权重（NaN/Inf/负数都当成 0）
        bad = ~np.isfinite(weights) | (weights < 0)
        if np.any(bad):
            weights = weights.copy()
            weights[bad] = 0.0
        
        # Fallback if weights sum to zero
        if weights.sum() <= 0:
            targets = rng.choice(candidates, size=m_eff, replace=False)
        else:
            probs = weights / weights.sum()
            targets = rng.choice(candidates, size=m_eff, replace=False, p=probs)

        # Add edges
        for t in targets:
            network.add_edge(int(v), int(t))

        # Update degree cache
        degrees[all_nodes == v] = float(m_eff)
        degrees[np.isin(all_nodes, targets)] += 1.0



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

    N = grid["N"]
    before_allocation = C_intake_grid.copy()

    # Pre-cache id -> (i, j)
    def id_to_ij_fast(node_id):
        return divmod(node_id, N)

    for component in nx.connected_components(network):

        # Convert component to array for faster processing
        nodes = np.fromiter(component, dtype=int)
        if nodes.size == 0:
            continue

        # Map nodes to grid indices
        ij = np.array([id_to_ij_fast(n) for n in nodes])
        ii, jj = ij[:, 0], ij[:, 1]

        # Gather values
        carbon = C_intake_grid[ii, jj].astype(float)
        cmax = max_carbon_intake_grid[ii, jj]

        carbon_sum = carbon.sum()

        # Sort by cmax (ascending)
        order = np.argsort(cmax)
        nodes_sorted = nodes[order]
        cmax_sorted = cmax[order]
        ii_sorted = ii[order]
        jj_sorted = jj[order]

        remaining = len(nodes_sorted)

        for k in range(remaining):
            avg_C = carbon_sum / (remaining - k)

            if cmax_sorted[k] < avg_C:
                val = cmax_sorted[k]
            else:
                val = avg_C

            C_intake_grid[ii_sorted[k], jj_sorted[k]] = val
            carbon_sum -= val

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
    death_prob = float(np.clip(env_stress, 0.0, 1.0))
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




def run_simulation(prob_seedling, scale_free_alpha, network_beta, env_stress, N, steps, seed=None, init_tree_density=0.0, init_biomass=1.0):
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
    

    # -----------------------
    # Main loop
    # -----------------------
    historical_data = []
    historical_transfer = []
    for t in range(steps):
        grid["step"] = t
        grow_seedlings(prob_seedling, network, grid, init_biomass)
        add_connections(network, grid, scale_free_alpha, beta=network_beta)
        carbon_intake, max_carbon_intake = calculate_C_intake(grid, env_stress)
        C_grid_former, C_grid_later = allocate_C_intake(network, grid, carbon_intake, max_carbon_intake)
        grid["biomass"] += C_grid_later
        check_survival(network, grid, env_stress)
        if t % 100 == 0:
            print(t)
    
    historical_data.append(grid.copy())
    historical_transfer.append(C_grid_later - C_grid_former)

    return network, grid, historical_data, historical_transfer

def analyze_hub_removal(network, grid, removal_fraction=0.05):
    side = grid["N"]

    net_hub = network.copy()
    grid_hub = copy.deepcopy(grid)

    net_rand = network.copy()
    grid_rand = copy.deepcopy(grid)

    nodes_by_degree = sorted(network.degree(), key=lambda x: x[1], reverse=True)
    num_to_remove = int(len(nodes_by_degree) * removal_fraction)
    hubs_to_remove = [n for n, d in nodes_by_degree[:num_to_remove]]

    all_nodes = list(network.nodes())
    random_to_remove = np.random.choice(all_nodes, num_to_remove, replace=False)

    def get_biomass_after_removal(net, g, nodes):
        for node in nodes:
            i, j = id_to_ij(node, side)
            remove_tree_at(net, g, i, j)
        return np.sum(g["biomass"])

    biomass_before = np.sum(grid["biomass"])
    biomass_hub = get_biomass_after_removal(net_hub, grid_hub, hubs_to_remove)
    biomass_rand = get_biomass_after_removal(net_rand, grid_rand, random_to_remove)

    return biomass_before, biomass_hub, biomass_rand


if __name__ == "__main__":

    fractions = [0, 0.02, 0.05, 0.1, 0.15, 0.2]
    hub_impact = []
    rand_impact = []

    net, grid, _, _ = run_simulation(0.2, 1, 0.8, 0.1, 50, 300, 42, 0.1)
    for f in fractions:
        _, b_hub, b_rand = analyze_hub_removal(net, grid, f)
        hub_impact.append(b_hub)
        rand_impact.append(b_rand)

    plt.figure(figsize=(8, 5))
    plt.plot(fractions, hub_impact, 'r-o', label='Remove Hubs (Targeted)')
    plt.plot(fractions, rand_impact, 'b--s', label='Remove Random Trees')
    plt.xlabel('Fraction of Trees Removed')
    plt.ylabel('Total Forest Biomass')
    plt.title('Impact of Hub Removal on Forest Biomass')
    plt.legend()
    plt.grid(True)
    plt.show()
# -----------------------------
# Run
# -----------------------------
# if __name__ == "__main__":
#     beta_values = [.2,.4,.6,.8]
#
#     stress_by_beta = []
#     biomass_by_beta = []
#
#     for b in beta_values:
#         stress = []
#         avg_biomass_list = []
#         for x in range(10):
#             stress.append(.1 * x)
#             network, grid, hist, trans = run_simulation(0.2, 1,b, .1*x, 50, 500, 3, 0.1)
#
#             tot_biomass = 0
#             for n in range(len(hist)):
#                 tot_biomass += np.sum(hist[n]["biomass"])
#             avg_biomass = tot_biomass / len(hist)
#             avg_biomass_list.append(avg_biomass)
#
#         stress_by_beta.append(stress)
#         biomass_by_beta.append(avg_biomass_list)
#
#
#     for b in range(len(beta_values)):
#         plt.plot(stress_by_beta[b][1:],biomass_by_beta[b][1:],marker='s',color='blue',label = "Beta = " + str(.1*b))
#
#     plt.yscale('log')
#     plt.legend()
#     plt.show()

