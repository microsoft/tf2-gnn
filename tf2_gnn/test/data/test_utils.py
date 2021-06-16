from typing import List, NamedTuple, Tuple, Union, Set

import numpy as np
import pytest
from tf2_gnn.data.utils import get_tied_edge_types, process_adjacency_lists


class TestInput(NamedTuple):
    adjacency_lists: List[List[Tuple[int, int]]]
    num_nodes: int
    add_self_loop_edges: bool
    tie_fwd_bkwd_edges: Union[bool, List[int]]
    self_loop_edge_type: int


class TestOutput(NamedTuple):
    adjacency_lists: List[np.ndarray]
    type_to_num_incoming_edges: np.ndarray


class TestCase(NamedTuple):
    test_input: TestInput
    expected_output: TestOutput


def create_test_input(
    add_self_loop_edges: bool,
    tie_fwd_bkwd_edges: Union[bool, List[int]],
    two_edge_types=False,
    self_loop_edge_type=0,
) -> TestInput:
    return TestInput(
        adjacency_lists=[[(0, 1)], [(1, 2)]] if two_edge_types else [[(0, 1), (1, 2)]],
        num_nodes=3,
        add_self_loop_edges=add_self_loop_edges,
        tie_fwd_bkwd_edges=tie_fwd_bkwd_edges,
        self_loop_edge_type=self_loop_edge_type,
    )


def create_test_output(
    adjacency_lists: List[List[Tuple[int, int]]], type_to_num_incoming_edges: List[List[int]]
) -> TestOutput:
    return TestOutput(
        adjacency_lists=[np.array(adj_list, dtype=np.int32) for adj_list in adjacency_lists],
        type_to_num_incoming_edges=np.array(type_to_num_incoming_edges),
    )


all_test_cases = [
    TestCase(
        test_input=create_test_input(add_self_loop_edges=False, tie_fwd_bkwd_edges=False),
        expected_output=create_test_output(
            adjacency_lists=[[(0, 1), (1, 2)], [(1, 0), (2, 1)]],
            type_to_num_incoming_edges=[[0, 1, 1], [1, 1, 0]],
        ),
    ),
    TestCase(
        test_input=create_test_input(add_self_loop_edges=False, tie_fwd_bkwd_edges=True),
        expected_output=create_test_output(
            adjacency_lists=[[(0, 1), (1, 2), (1, 0), (2, 1)]],
            type_to_num_incoming_edges=[[1, 2, 1]],
        ),
    ),
    TestCase(
        test_input=create_test_input(add_self_loop_edges=True, tie_fwd_bkwd_edges=False),
        expected_output=create_test_output(
            adjacency_lists=[[(0, 0), (1, 1), (2, 2)], [(0, 1), (1, 2)], [(1, 0), (2, 1)]],
            type_to_num_incoming_edges=[[1, 1, 1], [0, 1, 1], [1, 1, 0]],
        ),
    ),
    TestCase(
        test_input=create_test_input(add_self_loop_edges=True, tie_fwd_bkwd_edges=True),
        expected_output=create_test_output(
            adjacency_lists=[[(0, 0), (1, 1), (2, 2)], [(0, 1), (1, 2), (1, 0), (2, 1)]],
            type_to_num_incoming_edges=[[1, 1, 1], [1, 2, 1]],
        ),
    ),
    TestCase(
        test_input=create_test_input(
            add_self_loop_edges=True, tie_fwd_bkwd_edges=False, self_loop_edge_type=-1
        ),
        expected_output=create_test_output(
            adjacency_lists=[[(0, 1), (1, 2)], [(1, 0), (2, 1)], [(0, 0), (1, 1), (2, 2)]],
            type_to_num_incoming_edges=[[0, 1, 1], [1, 1, 0], [1, 1, 1]],
        ),
    ),
    TestCase(
        test_input=create_test_input(
            add_self_loop_edges=True, tie_fwd_bkwd_edges=True, self_loop_edge_type=-1
        ),
        expected_output=create_test_output(
            adjacency_lists=[[(0, 1), (1, 2), (1, 0), (2, 1)], [(0, 0), (1, 1), (2, 2)]],
            type_to_num_incoming_edges=[[1, 2, 1], [1, 1, 1]],
        ),
    ),
    TestCase(
        test_input=create_test_input(
            add_self_loop_edges=False, tie_fwd_bkwd_edges=[0], two_edge_types=True
        ),
        expected_output=create_test_output(
            adjacency_lists=[[(0, 1), (1, 0)], [(1, 2)], [(2, 1)]],
            type_to_num_incoming_edges=[[1, 1, 0], [0, 0, 1], [0, 1, 0]],
        ),
    ),
    TestCase(
        test_input=create_test_input(
            add_self_loop_edges=False, tie_fwd_bkwd_edges=[1], two_edge_types=True
        ),
        expected_output=create_test_output(
            adjacency_lists=[[(0, 1)], [(1, 2), (2, 1)], [(1, 0)]],
            type_to_num_incoming_edges=[[0, 1, 0], [0, 1, 1], [1, 0, 0]],
        ),
    ),
]


@pytest.mark.parametrize("test_case", all_test_cases)
def test_process_adjacency_lists(test_case: TestCase):
    inp = test_case.test_input
    adjacency_lists, type_to_num_incoming_edges = process_adjacency_lists(
        adjacency_lists=inp.adjacency_lists,
        num_nodes=inp.num_nodes,
        add_self_loop_edges=inp.add_self_loop_edges,
        tied_fwd_bkwd_edge_types=get_tied_edge_types(
            tie_fwd_bkwd_edges=inp.tie_fwd_bkwd_edges, num_fwd_edge_types=len(inp.adjacency_lists)
        ),
        self_loop_edge_type=inp.self_loop_edge_type,
    )

    out = test_case.expected_output

    assert len(adjacency_lists) == len(out.adjacency_lists)

    for adj_got, adj_expected in zip(adjacency_lists, out.adjacency_lists):
        assert np.array_equal(adj_got, adj_expected)

    assert np.array_equal(type_to_num_incoming_edges, out.type_to_num_incoming_edges)
