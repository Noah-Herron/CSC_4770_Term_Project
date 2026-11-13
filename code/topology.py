"""Broadcast topologies.

Rank 0 is the root (the master, which holds the data). Ranks 1..P-1 are workers.
Every builder returns `children`, a list where children[r] is the ranks that rank r
forwards the message to, in the order it sends to them.

Trees use the "clear the highest digit" rule: the parent of rank r is r with its
highest nonzero base-k digit set to zero. A rank's children are r + j * k^d for every
power k^d greater than r (j = 1..k-1). Sending to the smallest step first sends to the
largest subtree first, which gives the usual ceil(log_k P) rounds.
"""

TOPOLOGIES = ("binomial", "knomial", "chain", "flat")


def build_children(topology, num_nodes, k=4):
    if num_nodes < 1:
        raise ValueError("A broadcast needs at least one node (the root).")
    if topology == "flat":
        # The root sends to everyone itself (point-to-point from the owner, no forwarding)
        return [list(range(1, num_nodes))] + [[] for _ in range(num_nodes - 1)]
    if topology == "chain":
        # Each rank forwards to exactly one other rank
        return [[r + 1] if r + 1 < num_nodes else [] for r in range(num_nodes)]
    if topology == "binomial":
        k = 2
    elif topology == "knomial":
        if k < 2:
            raise ValueError("k-nomial trees need k >= 2.")
    else:
        raise ValueError(f"Unknown topology {topology!r}; choose from {', '.join(TOPOLOGIES)}.")

    children = [[] for _ in range(num_nodes)]
    for r in range(num_nodes):
        step = 1
        while step < num_nodes:
            if step > r:
                for j in range(1, k):
                    child = r + j * step
                    if child < num_nodes:
                        children[r].append(child)
            step *= k
    return children


def parents_from_children(children):
    """parents[r] is the rank that sends to r (None for the root). Checks it is a tree."""
    parents = [None] * len(children)
    for r, kids in enumerate(children):
        for child in kids:
            if not 0 < child < len(children):
                raise ValueError(f"Rank {r} has invalid child {child}.")
            if parents[child] is not None:
                raise ValueError(f"Rank {child} has two parents.")
            parents[child] = r
    missing = [r for r in range(1, len(children)) if parents[r] is None]
    if missing:
        raise ValueError(f"Ranks {missing} are never reached.")
    return parents


def depth(children):
    """Number of hops from the root to the deepest rank."""
    level = {0: 0}
    frontier = [0]
    while frontier:
        nxt = []
        for r in frontier:
            for child in children[r]:
                level[child] = level[r] + 1
                nxt.append(child)
        frontier = nxt
    if len(level) != len(children):
        raise ValueError("Topology does not reach every rank.")
    return max(level.values())