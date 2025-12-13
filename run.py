import numpy as np
import networkx as nx


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

def add_connections(network, grid, scale_free_alpha):
    """
    Add new mycorrizal network connections on the network layer using scale-free network building method
    """
    biomass = grid["biomass"]

    for node in list(network):
        preferential_attachment(network, biomass, node, scale_free_alpha)

def periodic_distance(p1, p2, shape):
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)

    delta = np.abs(p1-p2)

    delta = np.minimum(delta, shape-delta)

    return np.sqrt(np.sum(delta**2))



def preferential_attachment(network, forest, new_node, scale_free_alpha):
    """
    Docstring for preferential_attachment
    
    :param network: an nx graph
    :param node: an integer
    """

    nodes = list(network)
    side_len = len(forest)

    sum = 0

    p1 = id_to_ij(new_node)
    

    sum = 0
    cutoffs = []

    for n in nodes:
        p2 = id_to_ij(n)
        i2 = p2[0]
        j2 = p2[1]

        dist = periodic_distance(p1,p2,np.shape(forest))
        biomass = forest[i2,j2]
        val = np.exp(- scale_free_alpha * dist / biomass)

        cutoffs.append(val)
        sum += val

    cutoffs = [x / sum for x in cutoffs]
    r = np.random.rand()

    for x in range(len(cutoffs)):
        if(r < cutoffs[x]):
            network.add_edge(new_node,nodes[x])

            break
    
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
    max_carbon_intake = grid["carbon_intake"].copy()

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

    for s in subgraphs:
        carbon_sum = 0
        cmax_value = []
        for n in s:
            i,j = id_to_ij(n)
            carbon_sum += C_intake_grid[i,j]
            cmax_value.append(max_carbon_intake_grid[i,j])

        s_ordered, cmax_ordered = zip(*sorted(zip(s,cmax_value)))
        for n in range(s_ordered):
            avg_C = carbon_sum / (len(s_ordered) - n)
            i,j = id_to_ij(s_ordered[n])
            if(cmax_ordered[n] < avg_C):
                C_intake_grid[i,j] = cmax_ordered[n]
                carbon_sum -= cmax_ordered[n]
            else:
                C_intake_grid[i,j] = avg_C
                carbon_sum -= avg_C

    return C_intake_grid
                





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
    

    # -----------------------
    # Main loop
    # -----------------------
    for t in range(steps):
        grid["step"] = t

        # grow_seedlings(prob_seedling, network, grid, init_biomass)
        # add_connections(network, grid, scale_free_alpha)
        # calculate_C_intake(grid, env_stress)
        # growth_grid = allocate_C_intake(network, grid, grid["carbon_intake"])
        # grid["biomass"] += growth_grid
        # check_survival(network, grid, env_stress)

        pass

    return network, grid


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    pass