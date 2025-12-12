import numpy as np
import networkx as nx


# -----------------------------
# Helpers (grid + periodic BC)
# -----------------------------
def wrap(x, y, side):
    """Periodic boundary conditions."""
    return (x % side, y % side)


def all_positions(side):
    """Yield all (x, y) positions in the grid."""
    for y in range(side):
        for x in range(side):
            yield (x, y)


# -----------------------------
# Network layer
# -----------------------------
def connect_newborn_tree():
    """
    Connect newborn to up to existing nodes with probability `p_connect`.
    """


def redistribute_carbon():
    """
    Carbon redistribution on the network.
    """


# -----------------------------
# Spatial layer (state + step)
# -----------------------------
def step_forest_and_network(state, params, rng):
    """
    One iteration step:
      1) deaths
      2) births
      3) carbon production (proportional to biomass)
      4) network redistribution
      5) biomass update using redistributed carbon
    """
    side = params["side"]
    prob_birth = params["prob_birth"]
    prob_survival = params["prob_survival"]
    carbon_rate = params["carbon_rate"]

    p_connect = params["p_connect"]
    max_links = params["max_links"]
    diffusion_strength = params["diffusion_strength"]

    trees = state["trees"]  # dict: pos -> {"biomass":..., "carbon_intake":...}
    G = state["G"]          # networkx Graph

    # 1) Deaths
    for pos in list(trees.keys()):
        if rng.random() > prob_survival:
            del trees[pos]
            if pos in G:
                G.remove_node(pos)

    # 2) Births on empty cells
    newborns = []
    for pos in all_positions(side):
        if pos not in trees and rng.random() < prob_birth:
            trees[pos] = {"biomass": 1.0, "carbon_intake": 0.0}
            G.add_node(pos)
            newborns.append(pos)

    # Connect newborns (placeholder mechanism)
    if newborns:
        existing_positions = list(trees.keys())
        for new_pos in newborns:
            connect_newborn_tree(
                G, new_pos, existing_positions, rng,
                p_connect=p_connect, max_links=max_links
            )

    # 3) Carbon production (before redistribution)
    carbon_by_pos = {}
    for pos, tr in trees.items():
        carbon_by_pos[pos] = carbon_rate * tr["biomass"]

    # 4) Redistribute carbon via network layer
    redistributed = redistribute_carbon(
        G, carbon_by_pos, diffusion_strength=diffusion_strength
    )

    # 5) Biomass update (end of step): biomass += redistributed carbon_intake
    for pos, tr in trees.items():
        tr["carbon_intake"] = redistributed.get(pos, 0.0)
        tr["biomass"] += tr["carbon_intake"]


def run_simulation(params, steps=200, seed=0):
    """
    Run the demo simulation and return (state, history).
    """
    rng = np.random.default_rng(seed)

    state = {
        "trees": {},       # pos -> {"biomass": float, "carbon_intake": float}
        "G": nx.Graph(),   # mycorrhizal network
    }

    history = {
        "t": [],
        "n_trees": [],
        "n_edges": [],
        "total_biomass": [],
        "total_carbon": [],
    }

    for t in range(1, steps + 1):
        step_forest_and_network(state, params, rng)

        trees = state["trees"]
        G = state["G"]

        history["t"].append(t)
        history["n_trees"].append(len(trees))
        history["n_edges"].append(G.number_of_edges())
        history["total_biomass"].append(sum(tr["biomass"] for tr in trees.values()))
        history["total_carbon"].append(sum(tr["carbon_intake"] for tr in trees.values()))

    return state, history


# -----------------------------
# Demo run
# -----------------------------
if __name__ == "__main__":
    params = {
        # Grid layer
        "side": 30,
        "prob_birth": 0.02,
        "prob_survival": 0.98,

        # Tree physiology
        "carbon_rate": 0.05,  # carbon produced per step = carbon_rate * biomass

        # Network layer placeholders
        "p_connect": 0.15,
        "max_links": 2,
        "diffusion_strength": 0.30,
    }

    state, hist = run_simulation(params, steps=200, seed=42)

    print("Final trees:", hist["n_trees"][-1])
    print("Final edges:", hist["n_edges"][-1])
    print("Final total biomass:", hist["total_biomass"][-1])