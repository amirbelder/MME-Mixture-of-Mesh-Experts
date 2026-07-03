"""Random-walk sampling algorithms — vendored from the reference project's walks.py.

Every sampler returns ``(seq, jumps)`` where:
    seq   : (seq_len + 1,) int32 — vertex ids visited (seq[0] is the start).
    jumps : (seq_len + 1,) bool  — True on steps that were global/backtrack jumps.

This is the same interface the reference dataset builder uses; consume via
:mod:`mme.gate.walk_features.compose_walk_features` to match the paper's
``params.net_input`` composition and the reshape convention in
``train_val.py::train_step``.

The paper's default is :func:`get_seq_random_walk_random_global_jumps`.
"""

from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------
# Sequence samplers — each takes ``mesh_extra`` (dict with 'edges',
# 'n_vertices', optionally 'kdtree_query', 'vertices'), a starting vertex
# id ``f0``, and ``seq_len``. All return (seq, jumps).
# ------------------------------------------------------------------


def get_seq_random_walk_no_jumps(mesh_extra, f0, seq_len):
    """Pure walk with backtracking, no global jumps."""
    nbrs = mesh_extra["edges"]
    n_vertices = mesh_extra["n_vertices"]
    seq = np.zeros((seq_len + 1,), dtype=np.int32)
    jumps = np.zeros((seq_len + 1,), dtype=bool)
    visited = np.zeros((n_vertices + 1,), dtype=bool)
    visited[-1] = True
    visited[f0] = True
    seq[0] = f0
    jumps[0] = True
    backward_steps = 1
    for i in range(1, seq_len + 1):
        this_nbrs = nbrs[seq[i - 1]]
        candidates = [n for n in this_nbrs if not visited[n]]
        if candidates:
            to_add = np.random.choice(candidates)
            jump = False
        else:
            if i > backward_steps:
                to_add = seq[i - backward_steps - 1]
                backward_steps += 2
                jump = False
            else:
                to_add = np.random.randint(n_vertices)
                jump = True
        seq[i] = to_add
        jumps[i] = jump
        visited[to_add] = 1
    return seq, jumps


def get_seq_random_walk_random_global_jumps(mesh_extra, f0, seq_len):
    """Paper default. Backtracking, plus a global jump with probability 1/100."""
    nbrs = mesh_extra["edges"]
    n_vertices = mesh_extra["n_vertices"]
    seq = np.zeros((seq_len + 1,), dtype=np.int32)
    jumps = np.zeros((seq_len + 1,), dtype=bool)
    visited = np.zeros((n_vertices + 1,), dtype=bool)
    visited[-1] = True
    visited[f0] = True
    seq[0] = f0
    jumps[0] = True
    backward_steps = 1
    jump_prob = 1 / 100
    for i in range(1, seq_len + 1):
        this_nbrs = nbrs[seq[i - 1]]
        candidates = [n for n in this_nbrs if not visited[n]]
        jump_now = np.random.binomial(1, jump_prob)
        if candidates and not jump_now:
            to_add = candidates[np.random.randint(len(candidates))]
            jump = False
            backward_steps = 1
        else:
            if i > backward_steps and not jump_now:
                to_add = seq[i - backward_steps - 1]
                backward_steps += 2
                jump = True
            else:
                to_add = np.random.randint(n_vertices)
                jump = True
                visited[...] = 0
                visited[-1] = True
                backward_steps = 1
        visited[to_add] = 1
        seq[i] = to_add
        jumps[i] = jump
    return seq, jumps


def get_seq_random_walk_random_global_jumps_new(mesh_extra, f0, seq_len):
    """Newer variant: pop-from-stack backtracking instead of fixed backward-steps."""
    nbrs = mesh_extra["edges"]
    n_vertices = mesh_extra["n_vertices"]
    seq = np.zeros((seq_len + 1,), dtype=np.int32)
    jumps = np.zeros((seq_len + 1,), dtype=bool)
    visited = np.zeros((n_vertices + 1,), dtype=bool)
    visited[-1] = True
    visited[f0] = True
    seq[0] = f0
    jumps[0] = True
    backprop_stack: list = []
    jump_prob = 1 / 100
    for i in range(1, seq_len + 1):
        this_nbrs = nbrs[seq[i - 1]]
        candidates = [n for n in this_nbrs if not visited[n]]
        jump_now = np.random.binomial(1, jump_prob)
        if candidates and not jump_now:
            to_add = np.random.choice(candidates)
            jump = False
            backprop_stack.append(i - 1)
        else:
            if backprop_stack and not jump_now:
                to_add = seq[backprop_stack.pop()]
                jump = False
            else:
                backprop_stack = []
                to_add = np.random.randint(n_vertices)
                jump = True
                visited[...] = 0
                visited[-1] = True
        visited[to_add] = 1
        seq[i] = to_add
        jumps[i] = jump
    return seq, jumps


def get_seq_random_walk_constant_global_jumps(mesh_extra, f0, seq_len, k=10):
    """Deterministic jump every ``k`` steps."""
    nbrs = mesh_extra["edges"]
    n_vertices = mesh_extra["n_vertices"]
    seq = np.zeros((seq_len + 1,), dtype=np.int32)
    jumps = np.zeros((seq_len + 1,), dtype=bool)
    visited = np.zeros((n_vertices + 1,), dtype=bool)
    visited[-1] = True
    visited[f0] = True
    seq[0] = f0
    jumps[0] = True
    backprop_inds: list = []
    for i in range(1, seq_len + 1):
        this_nbrs = nbrs[seq[i - 1]]
        candidates = [n for n in this_nbrs if not visited[n]]
        jump_now = (i + 1) % k == 0
        if candidates and not jump_now:
            to_add = np.random.choice(candidates)
            jump = False
            backprop_inds.append(i - 1)
        else:
            if backprop_inds and not jump_now:
                to_add = seq[backprop_inds.pop()]
                jump = False
            else:
                to_add = np.random.randint(n_vertices)
                jump = True
                visited[...] = 0
                visited[-1] = True
                if jump_now:
                    backprop_inds = []
        visited[to_add] = 1
        seq[i] = to_add
        jumps[i] = jump
    return seq, jumps


def get_seq_random_walk_local_jumps(mesh_extra, f0, seq_len):
    """KDTree-based local jump when stuck. Needs mesh_extra['kdtree_query']."""
    n_vertices = mesh_extra["n_vertices"]
    kdtr = mesh_extra["kdtree_query"]
    seq = np.zeros((seq_len + 1,), dtype=np.int32)
    jumps = np.zeros((seq_len + 1,), dtype=bool)
    seq[0] = f0
    visited = np.zeros((n_vertices + 1,), dtype=bool)
    visited[-1] = True
    visited[f0] = True
    for i in range(1, seq_len + 1):
        to_consider = [n for n in kdtr[seq[i - 1]] if not visited[n]]
        if to_consider:
            seq[i] = np.random.choice(to_consider)
            jumps[i] = False
        else:
            seq[i] = np.random.randint(n_vertices)
            jumps[i] = True
            visited = np.zeros((n_vertices + 1,), dtype=bool)
            visited[-1] = True
        visited[seq[i]] = True
    return seq, jumps


# Registry so users (and MMEModel adapters) can select by string name,
# matching the reference ``params.walk_alg`` convention.
WALK_ALGORITHMS = {
    "no_jumps": get_seq_random_walk_no_jumps,
    "random_global_jumps": get_seq_random_walk_random_global_jumps,
    "random_global_jumps_new": get_seq_random_walk_random_global_jumps_new,
    "constant_global_jumps": get_seq_random_walk_constant_global_jumps,
    "local_jumps": get_seq_random_walk_local_jumps,
}


def get_walk_algorithm(name: str):
    """Look up a walk sampler by the same name the reference uses."""
    if name not in WALK_ALGORITHMS:
        raise KeyError(
            f"unknown walk_alg {name!r}; choose from {sorted(WALK_ALGORITHMS)}"
        )
    return WALK_ALGORITHMS[name]


# ------------------------------------------------------------------
# Convenience: sample N walks with a chosen algorithm.
# ------------------------------------------------------------------


def sample_paper_walks(
    mesh,
    num_walks: int,
    walk_len: int,
    walk_alg: str = "random_global_jumps",
    seed: int = None,
    constant_jump_k: int = 10,
):
    """Sample ``num_walks`` walks of length ``walk_len`` on ``mesh``.

    Returns:
        seqs  : (num_walks, walk_len + 1) int32 — vertex ids per walk.
        jumps : (num_walks, walk_len + 1) bool  — jump flags per walk.

    Both arrays include the starting vertex at position 0 (like the reference).
    The feature builders in :mod:`mme.gate.walk_features` handle the correct
    slicing when consuming these.
    """
    if seed is not None:
        np.random.seed(int(seed))

    mesh_extra = {
        "edges": mesh.vertex_neighbors,  # list of int arrays per vertex
        "n_vertices": mesh.num_vertices,
    }
    fn = get_walk_algorithm(walk_alg)

    seqs = np.zeros((num_walks, walk_len + 1), dtype=np.int32)
    jumps = np.zeros((num_walks, walk_len + 1), dtype=bool)
    for w in range(num_walks):
        f0 = int(np.random.randint(mesh.num_vertices))
        if walk_alg == "constant_global_jumps":
            s, j = fn(mesh_extra, f0, walk_len, k=constant_jump_k)
        else:
            s, j = fn(mesh_extra, f0, walk_len)
        seqs[w] = s
        jumps[w] = j
    return seqs, jumps
