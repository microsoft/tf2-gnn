from typing import List, Tuple

import numpy as np


Edge = Tuple[int, int]


def process_adjacency_lists(
    adjacency_lists: List[List[Edge]],
    num_nodes: int,
    add_self_loop_edges: bool,
    tie_fwd_bkwd_edges: bool,
) -> Tuple[List[np.ndarray], np.ndarray]:
    adjacency_lists = _add_backward_edges(adjacency_lists, tie_fwd_bkwd_edges)

    # Add self loops after adding backward edges to avoid adding loops twice.
    if add_self_loop_edges:
        adjacency_lists = _add_self_loop_edges(adjacency_lists, num_nodes)

    type_to_num_incoming_edges = _compute_type_to_num_inedges(
        adjacency_lists=adjacency_lists, num_nodes=num_nodes
    )

    return _convert_adjacency_lists_to_numpy_arrays(adjacency_lists), type_to_num_incoming_edges


def _add_self_loop_edges(adjacency_lists: List[List[Edge]], num_nodes: int) -> List[List[Edge]]:
    self_loops = [(i, i) for i in range(num_nodes)]
    return [self_loops] + adjacency_lists


def _add_backward_edges(
    adjacency_lists: List[List[Edge]], tie_fwd_bkwd_edges: bool
) -> List[List[Edge]]:
    flipped_adjacency_lists = [
        [(dest, src) for (src, dest) in adjacency_list] for adjacency_list in adjacency_lists
    ]

    if tie_fwd_bkwd_edges:
        return [
            adj + adj_flipped
            for (adj, adj_flipped) in zip(adjacency_lists, flipped_adjacency_lists)
        ]
    else:
        return adjacency_lists + flipped_adjacency_lists


def _compute_type_to_num_inedges(adjacency_lists: List[List[Edge]], num_nodes: int) -> np.ndarray:
    num_edge_types = len(adjacency_lists)
    type_to_num_incoming_edges = np.zeros(shape=(num_edge_types, num_nodes))

    for edge_type, edges in enumerate(adjacency_lists):
        for _, dest in edges:
            type_to_num_incoming_edges[edge_type, dest] += 1

    return type_to_num_incoming_edges


def _convert_adjacency_lists_to_numpy_arrays(adjacency_lists: List[List[Edge]]) -> List[np.ndarray]:
    return [
        np.array(adj_list, dtype=np.int32)
        if len(adj_list) > 0
        else np.zeros(shape=(0, 2), dtype=np.int32)
        for adj_list in adjacency_lists
    ]
