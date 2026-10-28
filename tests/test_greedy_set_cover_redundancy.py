#!/usr/bin/env python3
"""Unit tests for greedy set cover redundancy filtering."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from greedy_set_cover_redundancy import (
    GroupData,
    assign_removed_to_representatives,
    coverage_metrics,
    filter_all_groups,
    filter_group,
    greedy_set_cover_indices,
    old_method_remove_from_pairs,
)


def _star_matrix(n_leaves: int) -> tuple[list[str], np.ndarray]:
    """Center connected to all leaves; leaves not connected to each other."""
    n = n_leaves + 1
    names = ["center"] + [f"leaf_{i:02d}" for i in range(n_leaves)]
    matrix = np.eye(n, dtype=np.float64)
    for i in range(1, n):
        matrix[0, i] = matrix[i, 0] = 0.95
    return names, matrix


class TestGreedySetCoverCore(unittest.TestCase):
    def test_no_redundant_pairs_keeps_all(self):
        names = ["a", "b", "c"]
        matrix = np.eye(3)
        group = GroupData("g", names, matrix)
        result = filter_group(group, tau=0.90)
        self.assertEqual(sorted(result.kept), names)
        self.assertEqual(result.removed, [])
        self.assertEqual(set(result.kept) | set(result.removed), set(names))
        self.assertFalse(set(result.kept) & set(result.removed))

    def test_full_clique_keeps_one(self):
        names = ["a", "b", "c", "d"]
        matrix = np.full((4, 4), 0.99)
        np.fill_diagonal(matrix, 1.0)
        group = GroupData("g", names, matrix)
        result = filter_group(group, tau=0.90)
        self.assertEqual(len(result.kept), 1)
        self.assertEqual(len(result.removed), 3)
        self.assertEqual(set(result.kept) | set(result.removed), set(names))
        self.assertFalse(set(result.kept) & set(result.removed))
        for rem, asg in result.assignments.items():
            self.assertEqual(asg["representative"], result.kept[0])
            self.assertGreaterEqual(asg["similarity"], 0.90)

    def test_coverage_invariant_every_original_covered(self):
        names, matrix = _star_matrix(5)
        group = GroupData("star", names, matrix)
        result = filter_group(group, tau=0.90)
        kept = set(result.kept)
        for name in names:
            if name in kept:
                continue
            self.assertIn(name, result.assignments)
            self.assertGreaterEqual(result.assignments[name]["similarity"], 0.90)

        metrics = coverage_metrics([group], kept, 0.90)
        self.assertEqual(metrics["coverage_pct"], 100.0)

    def test_kept_subset_of_original(self):
        names, matrix = _star_matrix(4)
        group = GroupData("star", names, matrix)
        result = filter_group(group, tau=0.90)
        self.assertTrue(set(result.kept).issubset(set(names)))
        self.assertTrue(set(result.removed).issubset(set(names)))

    def test_partition_union_disjoint(self):
        names, matrix = _star_matrix(6)
        group = GroupData("star", names, matrix)
        result = filter_group(group, tau=0.90)
        self.assertEqual(set(result.kept) & set(result.removed), set())
        self.assertEqual(set(result.kept) | set(result.removed), set(names))

    def test_deterministic(self):
        names, matrix = _star_matrix(8)
        # Add slight asymmetry so tie-breaks matter.
        matrix[0, 1] = matrix[1, 0] = 0.96
        matrix[0, 2] = matrix[2, 0] = 0.94
        g = GroupData("star", names, matrix)
        r1 = filter_group(g, tau=0.90)
        r2 = filter_group(g, tau=0.90)
        self.assertEqual(r1.kept, r2.kept)
        self.assertEqual(r1.removed, r2.removed)
        self.assertEqual(r1.selection_order, r2.selection_order)

    def test_groups_are_independent(self):
        # Cross-group similarities would be ignored because groups are separate matrices.
        a = GroupData(
            "ga",
            ["a1", "a2"],
            np.array([[1.0, 0.99], [0.99, 1.0]]),
        )
        b = GroupData(
            "gb",
            ["b1", "b2"],
            np.array([[1.0, 0.50], [0.50, 1.0]]),
        )
        result = filter_all_groups([a, b], threshold=90)
        by_group = {g.group: g for g in result.groups}
        self.assertEqual(len(by_group["ga"].kept), 1)
        self.assertEqual(sorted(by_group["gb"].kept), ["b1", "b2"])

    def test_removed_has_representative_above_threshold(self):
        names = ["x", "y", "z"]
        matrix = np.array(
            [
                [1.0, 0.93, 0.50],
                [0.93, 1.0, 0.50],
                [0.50, 0.50, 1.0],
            ]
        )
        result = filter_group(GroupData("g", names, matrix), tau=0.90)
        for rem, asg in result.assignments.items():
            self.assertGreaterEqual(asg["similarity"], 0.90)
            self.assertIn(asg["representative"], result.kept)

    def test_assignment_picks_max_similarity(self):
        names = ["k1", "k2", "r"]
        matrix = np.array(
            [
                [1.0, 0.1, 0.91],
                [0.1, 1.0, 0.97],
                [0.91, 0.97, 1.0],
            ]
        )
        kept_idx = {0, 1}
        asg = assign_removed_to_representatives(names, matrix, kept_idx, [2], 0.90)
        self.assertEqual(asg["r"]["representative"], "k2")
        self.assertAlmostEqual(asg["r"]["similarity"], 0.97)

    def test_star_differs_from_old_frequency_method(self):
        """
        Star graph: frequency removal deletes the hub; set cover keeps the hub.

        Old: remove highest-degree vertex (center) → keep all leaves.
        New: center covers everyone → keep center only.
        """
        names, matrix = _star_matrix(5)
        tau = 0.90
        pairs = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if matrix[i, j] >= tau:
                    pairs.append((names[i], names[j]))

        old_removed = old_method_remove_from_pairs(pairs)
        old_kept = sorted(set(names) - set(old_removed))
        new = filter_group(GroupData("star", names, matrix), tau=tau)

        self.assertEqual(old_removed, ["center"])
        self.assertEqual(old_kept, sorted(n for n in names if n != "center"))
        self.assertEqual(new.kept, ["center"])
        self.assertEqual(sorted(new.removed), sorted(n for n in names if n != "center"))

    def test_already_covered_non_rep_can_still_be_selected(self):
        """
        Classic set-cover behaviour: a covered non-representative may later
        become a representative if that maximises newly covered elements.

        Graph (tau edges):
          A—B—C—D   plus   E—C
        Threshold edges all 0.95.
        One valid greedy path can select B (covers A,B,C), then D or E, etc.
        Construct a case where after picking A (covers A,B), C is covered via...
        Better explicit construction:

          Nodes: R1, X, Y, Z
          Edges: R1-X, X-Y, X-Z  (X is the useful second pick after somehow...)

        Simpler forced scenario with asymmetric covers via threshold:
          Use complete bipartite-ish structure encoded in similarities.
        """
        # Names ordered so label tie-break is controlled.
        names = ["a", "b", "c", "d"]
        # Edges >= 0.9: a-b, b-c, c-d. Path of length 3.
        matrix = np.eye(4)
        matrix[0, 1] = matrix[1, 0] = 0.95
        matrix[1, 2] = matrix[2, 1] = 0.95
        matrix[2, 3] = matrix[3, 2] = 0.95

        selected, covers = greedy_set_cover_indices(matrix, 0.90, labels=names)
        # First pick should be b or c (cover size 3). Label 'b' < 'c' and both
        # have mean sim 0.95 over two others → 'b' wins.
        self.assertEqual(names[selected[0]], "b")
        # After b selected, uncovered = {d}. Candidates:
        # a: covers {a,b} ∩ U = empty
        # c: covers {b,c,d} ∩ U = {d}
        # d: covers {c,d} ∩ U = {d}
        # c was already covered by b but not selected; still eligible → chosen by label? 
        # mean for c over {d}=0.95; for d over empty others in newly={d} → mean -1
        # so c wins. This proves covered-non-rep can be selected.
        self.assertIn("c", [names[i] for i in selected])
        self.assertTrue(set(selected) >= {1})  # b index
        # c index is 2; should be selected even though covered by b first.
        self.assertIn(2, selected)


class TestMatByteIdentity(unittest.TestCase):
    def test_copy_preserves_bytes(self):
        from build_filtered_mat_from_removing_json import copy_or_link

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src = tmp_path / "manoeuvre.mat"
            payload = b"MATLAB-fake-bytes-" + hashlib.sha256(b"x").digest()
            src.write_bytes(payload)
            dst = tmp_path / "out" / "manoeuvre.mat"
            dst.parent.mkdir()
            copy_or_link(src, dst, "copy")
            self.assertEqual(src.read_bytes(), dst.read_bytes())


class TestTeslaMatricesSmoke(unittest.TestCase):
    """Optional smoke test against real text_vlm_tesla matrices if present."""

    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.matrices = cls.root / "cosine_similarity_matrices" / "text_vlm_tesla"
        cls.available = (
            cls.matrices.is_dir()
            and any(cls.matrices.glob("text_vlm_tesla_cosine_similarity_matrix_*_labels.json"))
        )

    def test_full_partition_and_coverage_on_real_data(self):
        if not self.available:
            self.skipTest("text_vlm_tesla matrices not available")
        from greedy_set_cover_redundancy import load_group_matrices

        groups = load_group_matrices(self.matrices, "text_vlm_tesla")
        self.assertEqual(sum(len(g.manoeuvres) for g in groups), 458)
        result = filter_all_groups(groups, 95)
        universe = {n for g in groups for n in g.manoeuvres}
        self.assertEqual(set(result.kept) | set(result.removed), universe)
        self.assertFalse(set(result.kept) & set(result.removed))
        metrics = coverage_metrics(groups, set(result.kept), 0.95)
        self.assertEqual(metrics["coverage_pct"], 100.0)
        # Determinism on real data
        result2 = filter_all_groups(groups, 95)
        self.assertEqual(result.kept, result2.kept)


if __name__ == "__main__":
    unittest.main()
