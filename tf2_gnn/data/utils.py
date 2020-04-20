from typing import List, Set, Tuple, Union

import numpy as np

Edge = Tuple[int, int]


def process_adjacency_lists(
    adjacency_lists: List[List[Edge]],
    num_nodes: int,
    add_self_loop_edges: bool,
    tied_fwd_bkwd_edge_types: Set[int],
) -> Tuple[List[np.ndarray], np.ndarray]:
    adjacency_lists = _add_backward_edges(adjacency_lists, tied_fwd_bkwd_edge_types)

    # Add self loops after adding backward edges to avoid adding loops twice.
    if add_self_loop_edges:
        adjacency_lists = _add_self_loop_edges(adjacency_lists, num_nodes)

    type_to_num_incoming_edges = _compute_type_to_num_inedges(
        adjacency_lists=adjacency_lists, num_nodes=num_nodes
    )

    return _convert_adjacency_lists_to_numpy_arrays(adjacency_lists), type_to_num_incoming_edges


def get_tied_edge_types(
    tie_fwd_bkwd_edges: Union[bool, Set[int]], num_fwd_edge_types: int
) -> Set[int]:
    if isinstance(tie_fwd_bkwd_edges, set):
        return tie_fwd_bkwd_edges
    elif tie_fwd_bkwd_edges:
        return set(range(num_fwd_edge_types))
    else:
        return {}


def compute_number_of_edge_types(
    tied_fwd_bkwd_edge_types: Set[int], num_fwd_edge_types: int, add_self_loop_edges: bool
) -> int:
    """Computes the number of edge types after adding backward edges and possibly self loops."""
    return 2 * num_fwd_edge_types - len(tied_fwd_bkwd_edge_types) + int(add_self_loop_edges)


def _add_self_loop_edges(adjacency_lists: List[List[Edge]], num_nodes: int) -> List[List[Edge]]:
    self_loops = [(i, i) for i in range(num_nodes)]
    return [self_loops] + adjacency_lists


def _add_backward_edges(
    adjacency_lists: List[List[Edge]], tied_fwd_bkwd_edge_types: Set[int]
) -> List[List[Edge]]:
    # Make sure the output will contain newly created lists.
    new_adjacency_lists = [adj_list.copy() for adj_list in adjacency_lists]

    for edge_type in range(len(adjacency_lists)):
        flipped_adjacency_list = [(dest, src) for (src, dest) in adjacency_lists[edge_type]]

        if edge_type in tied_fwd_bkwd_edge_types:
            new_adjacency_lists[edge_type] += flipped_adjacency_list
        else:
            new_adjacency_lists.append(flipped_adjacency_list)

    return new_adjacency_lists


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
