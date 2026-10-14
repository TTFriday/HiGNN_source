# -*- coding: utf-8 -*-
"""Standalone evaluation regression tests; run with unittest, without training/data."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluate import (
    binary_metrics,
    binary_metrics_by_phenotype,
    frequency_group_summary,
    ranking_metrics,
    split_by_frequency,
    standard_ranking_metrics,
)


def pairs(*items):
    return torch.tensor(items, dtype=torch.long).reshape(-1, 2).T.contiguous()


EMPTY = pairs()


def historical_ranking_reference(S, train_pos, test_pos, ks=(10, 50)):
    """Frozen original calculation, including its untruncated historical DRS key."""
    S_np = S.detach().cpu().numpy()
    tp_by_p, tr_by_p = {}, {}
    for g, p in test_pos.cpu().numpy().T:
        tp_by_p.setdefault(int(p), []).append(int(g))
    for g, p in train_pos.cpu().numpy().T:
        tr_by_p.setdefault(int(p), []).append(int(g))
    rr, ndcg, hits = [], [], {k: [] for k in ks}
    for p, gs in tp_by_p.items():
        s = S_np[:, p].copy()
        blocked = tr_by_p.get(p, [])
        if blocked:
            s[np.array(blocked)] = -np.inf
        order = np.argsort(-s)
        rank_of = np.empty(len(order), dtype=np.int64)
        rank_of[order] = np.arange(len(order))
        for g in gs:
            r = int(rank_of[g])
            rr.append(1.0 / (r + 1))
            ndcg.append(1.0 / np.log2(r + 2))
            for k in ks:
                hits[k].append(1.0 if r < k else 0.0)
    out = {"MRR": float(np.mean(rr)), "NDCG@10": float(np.mean(ndcg))}
    for k in ks:
        out[f"Hits@{k}"] = float(np.mean(hits[k]))
    return out


class StandardRankingTests(unittest.TestCase):
    def test_sigmoid_saturation_keeps_logit_order(self):
        logits = torch.tensor([[80.0], [100.0], [90.0]], requires_grad=True)
        self.assertTrue(torch.equal(torch.sigmoid(logits), torch.ones_like(logits)))
        test = pairs((2, 0))
        actual = standard_ranking_metrics(logits, EMPTY, EMPTY, test, ks=(1, 2))
        saturated = standard_ranking_metrics(torch.sigmoid(logits), EMPTY, EMPTY, test, ks=(1, 2))
        self.assertEqual(actual["pair_MRR"], 0.5)
        self.assertEqual(actual["pair_Hits@2"], 1.0)
        self.assertAlmostEqual(saturated["pair_MRR"], 1 / 3)
        self.assertEqual(saturated["pair_Hits@2"], 0.0)
        self.assertEqual(ranking_metrics(torch.sigmoid(logits), EMPTY, test),
                         historical_ranking_reference(torch.sigmoid(logits), EMPTY, test))

    def test_train_and_validation_are_removed_from_candidates(self):
        logits = torch.tensor([[100.0], [90.0], [80.0], [70.0]])
        train, valid, test = pairs((0, 0)), pairs((1, 0)), pairs((2, 0))
        with patch("src.evaluate.np.lexsort", wraps=np.lexsort) as sort:
            actual = standard_ranking_metrics(logits, train, valid, test)
        candidate_ids, candidate_scores = sort.call_args.args[0]
        np.testing.assert_array_equal(candidate_ids, [2, 3])
        np.testing.assert_array_equal(candidate_scores, [-80, -70])
        self.assertEqual(actual["pair_MRR"], 1.0)
        self.assertEqual(actual["macro_NDCG@50"], 1.0)
        unfiltered_valid = standard_ranking_metrics(logits, train, EMPTY, test)
        self.assertEqual(unfiltered_valid["pair_MRR"], 0.5)

    def test_filter_is_specific_to_the_phenotype(self):
        logits = np.array([[4, 4], [3, 3], [2, 2]], dtype=np.float32)
        actual = standard_ranking_metrics(logits, pairs((0, 0)), pairs((1, 0)),
                                          pairs((0, 1), (2, 0)), ks=(1,))
        self.assertEqual(actual["pair_MRR"], 1.0)
        self.assertEqual(actual["n_test_phenotypes"], 2)

    def test_ties_use_ascending_gene_index(self):
        logits = np.ones((6, 1), dtype=np.float32)
        actual = standard_ranking_metrics(logits, pairs((1, 0)), EMPTY,
                                          pairs((2, 0)), ks=(1, 2))
        self.assertEqual(actual["pair_MRR"], 0.5)
        self.assertEqual(actual["pair_Hits@1"], 0.0)
        self.assertEqual(actual["pair_Hits@2"], 1.0)

    def test_fp64_scores_are_ranked_after_fp32_conversion(self):
        logits = np.array([[1.0], [1.0 + 1e-9]], dtype=np.float64)
        actual = standard_ranking_metrics(logits, EMPTY, EMPTY, pairs((1, 0)), ks=(1,))
        self.assertEqual(actual["pair_MRR"], 0.5)
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                actual = standard_ranking_metrics(torch.tensor([[1], [2]], dtype=dtype),
                                                  EMPTY, EMPTY, pairs((1, 0)))
                self.assertEqual(actual["pair_MRR"], 1.0)

    def test_multiple_positives_binary_ndcg_and_macro_weighting(self):
        # Phenotype 0: positive ranks 1,3,5; phenotype 1: positive rank 2.
        logits = np.repeat(np.arange(5, 0, -1, dtype=np.float32)[:, None], 2, axis=1)
        test = pairs((0, 0), (2, 0), (4, 0), (1, 1))
        actual = standard_ranking_metrics(logits, EMPTY, EMPTY, test, ks=(2, 3, 10, 50))
        d2, d5 = 1 / np.log2(3), 1 / np.log2(6)
        self.assertAlmostEqual(actual["pair_MRR"], (1 + 1 / 3 + 1 / 5 + 1 / 2) / 4)
        self.assertEqual(actual["macro_MRR_first"], 0.75)
        self.assertEqual(actual["pair_Hits@2"], 0.5)
        self.assertAlmostEqual(actual["macro_Recall@2"], (1 / 3 + 1) / 2)
        self.assertAlmostEqual(actual["macro_NDCG@2"], (1 / (1 + d2) + d2) / 2)
        self.assertEqual(actual["pair_Hits@3"], 0.75)
        self.assertAlmostEqual(actual["macro_Recall@3"], (2 / 3 + 1) / 2)
        self.assertAlmostEqual(actual["macro_NDCG@3"], (1.5 / (1.5 + d2) + d2) / 2)
        self.assertAlmostEqual(actual["macro_NDCG@10"], ((1.5 + d5) / (1.5 + d2) + d2) / 2)
        self.assertEqual(actual["macro_NDCG@10"], actual["macro_NDCG@50"])
        self.assertEqual(actual["macro_Recall@50"], 1.0)
        self.assertEqual(actual["n_test_phenotypes"], 2)
        self.assertEqual(actual["n_test_pairs"], 4)
        reordered = standard_ranking_metrics(logits, EMPTY, EMPTY, test.flip(1), ks=(2, 3, 10, 50))
        self.assertEqual(actual, reordered)

    def test_ndcg_cutoff_misses_and_perfect_multi_positive(self):
        scores = np.arange(4, 0, -1, dtype=np.float32)[:, None]
        miss = standard_ranking_metrics(scores, EMPTY, EMPTY, pairs((3, 0)), ks=(1, 2))
        self.assertEqual(miss["macro_Recall@2"], 0.0)
        self.assertEqual(miss["macro_NDCG@2"], 0.0)
        perfect = standard_ranking_metrics(scores, EMPTY, EMPTY,
                                           pairs((0, 0), (1, 0)), ks=(1, 2, 50))
        for k in (1, 2, 50):
            self.assertEqual(perfect[f"macro_NDCG@{k}"], 1.0)
        self.assertEqual(perfect["macro_Recall@1"], 0.5)

    def test_empty_returns_none_and_zero_without_warnings(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for scores in (np.empty((0, 0)), np.ones((3, 2))):
                actual = standard_ranking_metrics(scores, [], EMPTY, [], ks=(10, 50))
                self.assertEqual(actual["n_test_phenotypes"], 0)
                self.assertEqual(actual["n_test_pairs"], 0)
                self.assertTrue(all(value is None for key, value in actual.items()
                                    if not key.startswith("n_")))

    def test_test_overlap_with_train_or_validation_raises(self):
        scores = np.ones((3, 2))
        for train, valid in ((pairs((1, 0)), EMPTY), (EMPTY, pairs((1, 0)))):
            with self.subTest(train=train, valid=valid), self.assertRaisesRegex(ValueError, "overlap"):
                standard_ranking_metrics(scores, train, valid, pairs((1, 0)))
        # Train/validation overlap with each other is simply a set union.
        actual = standard_ranking_metrics(scores, pairs((0, 0)), pairs((0, 0)), pairs((1, 0)))
        self.assertEqual(actual["pair_MRR"], 1.0)

    def test_invalid_cutoffs(self):
        for k in (0, -1, 1.5, np.inf, True):
            with self.subTest(k=k), self.assertRaises(ValueError):
                standard_ranking_metrics(np.ones((2, 1)), EMPTY, EMPTY, EMPTY, ks=(k,))


class BinaryAndFrequencyTests(unittest.TestCase):
    def test_macro_auroc_auprc_not_weighted_by_pair_count(self):
        scores = np.array([[2, 0], [1, 3], [0, 2], [0, 1]], dtype=np.float32)
        actual = binary_metrics_by_phenotype(scores, pairs((0, 0), (0, 1)),
                                            pairs((1, 0), (1, 1), (2, 1), (3, 1)))
        self.assertEqual(actual["macro_AUROC"], 0.5)
        self.assertEqual(actual["macro_AUPRC"], (1 + 1 / 4) / 2)
        self.assertEqual(actual["n_phenotypes"], 2)
        self.assertEqual(actual["n_pairs"], 6)

    def test_macro_auprc_at_constant_scores_is_macro_prevalence(self):
        # Eligible prevalence: p0=1/4, p1=2/3; p2/p3 have just one class.
        scores = torch.full((4, 4), 0.5)
        pos = pairs((0, 0), (0, 1), (1, 1), (0, 2))
        neg = pairs((1, 0), (2, 0), (3, 0), (2, 1), (0, 3))
        actual = binary_metrics_by_phenotype(scores, pos, neg)
        self.assertEqual(actual["macro_AUROC"], 0.5)
        self.assertAlmostEqual(actual["macro_AUPRC"], (1 / 4 + 2 / 3) / 2)
        self.assertNotAlmostEqual(actual["macro_AUPRC"], 3 / 7)
        self.assertEqual(actual["n_phenotypes"], 2)
        self.assertEqual(actual["n_pairs"], 7)

    def test_binary_empty_or_single_class_phenotypes(self):
        expected = {"macro_AUROC": None, "macro_AUPRC": None, "n_phenotypes": 0, "n_pairs": 0}
        for pos, neg in ((EMPTY, EMPTY), (pairs((0, 0)), EMPTY),
                         (EMPTY, pairs((0, 1))), (pairs((0, 0)), pairs((0, 1)))):
            with self.subTest(pos=pos, neg=neg):
                self.assertEqual(binary_metrics_by_phenotype(np.ones((2, 2)), pos, neg), expected)
        self.assertEqual(binary_metrics_by_phenotype(np.empty((0, 0)), [], []), expected)

    def test_binary_label_overlap_raises(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            binary_metrics_by_phenotype(np.ones((2, 1)), pairs((0, 0)), pairs((0, 0)))

    def test_repeated_sampled_negatives_preserve_multiplicity(self):
        pos, neg = pairs((0, 0)), pairs((1, 0), (1, 0), (2, 0))
        result = binary_metrics_by_phenotype(np.ones((3, 1)), pos, neg)
        self.assertEqual(result["n_pairs"], 4)
        self.assertEqual(result["macro_AUPRC"], 0.25)
        group = frequency_group_summary(EMPTY, pos, neg)["tail"]
        self.assertEqual(group["n_test_negatives"], 3)
        self.assertEqual(group["prevalence"], 0.25)

    def test_frequency_counts_and_group_pair_prevalence(self):
        train = pairs((10, 0), (11, 0), (10, 1), (10, 4), (11, 4))
        pos = pairs((0, 0), (0, 1), (1, 1), (0, 2))
        neg = pairs((1, 0), (2, 0), (3, 0), (2, 1), (0, 3))
        actual = frequency_group_summary(train, pos, neg, n_min=2)
        self.assertEqual(actual["head"], {"n_phenotypes": 1, "n_test_positives": 1,
                                        "n_test_negatives": 3, "prevalence": 0.25})
        self.assertEqual(actual["tail"], {"n_phenotypes": 3, "n_test_positives": 3,
                                        "n_test_negatives": 2, "prevalence": 0.6})
        # A threshold of zero includes phenotypes with no training positives in head.
        all_head = frequency_group_summary(train, pos, neg, n_min=0)
        self.assertEqual(all_head["head"]["n_phenotypes"], 4)
        self.assertAlmostEqual(all_head["head"]["prevalence"], 4 / 9)
        self.assertEqual(all_head["tail"]["n_phenotypes"], 0)
        self.assertIsNone(all_head["tail"]["prevalence"])

    def test_frequency_empty_and_single_class_groups(self):
        empty_group = {"n_phenotypes": 0, "n_test_positives": 0,
                       "n_test_negatives": 0, "prevalence": None}
        self.assertEqual(frequency_group_summary([], [], []),
                         {"head": empty_group, "tail": empty_group})
        train = pairs((2, 0))
        self.assertEqual(frequency_group_summary(train, EMPTY, EMPTY, n_min=1),
                         {"head": empty_group, "tail": empty_group})
        actual = frequency_group_summary(train, pairs((0, 0)), pairs((0, 1)), n_min=1)
        self.assertEqual(actual["head"]["prevalence"], 1.0)
        self.assertEqual(actual["tail"]["prevalence"], 0.0)

    def test_frequency_overlap_and_invalid_threshold(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            frequency_group_summary(EMPTY, pairs((0, 0)), pairs((0, 0)))
        for n_min in (-1, np.inf, np.nan, 1.5, True):
            with self.subTest(n_min=n_min), self.assertRaises(ValueError):
                frequency_group_summary(EMPTY, EMPTY, EMPTY, n_min=n_min)


class ValidationAndHistoricalTests(unittest.TestCase):
    def test_nonfinite_or_nonreal_scores_rejected_even_without_test_pairs(self):
        bad_scores = [np.array([[value]]) for value in (np.nan, np.inf, -np.inf, 1e100)]
        bad_scores += [torch.tensor([[float("nan")]]), np.array([[1 + 2j]]),
                       torch.tensor([[1 + 2j]]), np.ones(3), np.array([["not a score"]])]
        for scores in bad_scores:
            with self.subTest(scores=scores):
                with self.assertRaises(ValueError):
                    standard_ranking_metrics(scores, EMPTY, EMPTY, EMPTY)
                with self.assertRaises(ValueError):
                    binary_metrics_by_phenotype(scores, EMPTY, EMPTY)

    def test_invalid_pairs_rejected(self):
        malformed = [np.array([[np.nan], [0]]), np.array([[np.inf], [0]]),
                     np.array([[0.5], [0]]), np.array([[-1], [0]]),
                     np.ones((3, 1), dtype=np.int64), pairs((0, 0), (0, 0))]
        for bad in malformed:
            with self.subTest(pairs=bad):
                for position in range(3):
                    args = [EMPTY, EMPTY, EMPTY]
                    args[position] = bad
                    with self.assertRaises(ValueError):
                        standard_ranking_metrics(np.ones((2, 2)), *args)
                    if position != 2 or not np.array_equal(bad, pairs((0, 0), (0, 0))):
                        with self.assertRaises(ValueError):
                            frequency_group_summary(*args)
                with self.assertRaises(ValueError):
                    binary_metrics_by_phenotype(np.ones((2, 2)), bad, EMPTY)
        for bad in (pairs((2, 0)), pairs((0, 2))):
            with self.subTest(out_of_bounds=bad):
                with self.assertRaises(ValueError):
                    standard_ranking_metrics(np.ones((2, 2)), EMPTY, EMPTY, bad)
                with self.assertRaises(ValueError):
                    binary_metrics_by_phenotype(np.ones((2, 2)), EMPTY, bad)

    def test_historical_ranking_exact_values_and_keys_are_unchanged(self):
        scores = torch.sigmoid(torch.tensor([[80, -1], [90, 0], [100, 2], [0, 2]], dtype=torch.float32))
        train = pairs((0, 0), (3, 1))
        # Historical overlap stays permitted; a blocked gene still has a tail rank.
        test = pairs((1, 0), (2, 0), (0, 1), (3, 1))
        for ks in ((10, 50), (1, 2, 3)):
            self.assertEqual(ranking_metrics(scores, train, test, ks),
                             historical_ranking_reference(scores, train, test, ks))

    def test_historical_ndcg10_is_untruncated_drs(self):
        scores = torch.arange(11, 0, -1, dtype=torch.float32)[:, None]
        actual = ranking_metrics(scores, EMPTY, pairs((10, 0)))
        self.assertAlmostEqual(actual["MRR"], 1 / 11)
        self.assertAlmostEqual(actual["NDCG@10"], 1 / np.log2(12))
        self.assertEqual(actual["Hits@10"], 0.0)
        standard = standard_ranking_metrics(scores, EMPTY, EMPTY, pairs((10, 0)))
        self.assertEqual(standard["macro_NDCG@10"], 0.0)

    def test_historical_ranking_empty_remains_nan(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            actual = ranking_metrics(torch.ones((2, 1)), EMPTY, EMPTY)
        self.assertTrue(all(np.isnan(value) for value in actual.values()))

    def test_historical_binary_and_frequency_remain_pooled(self):
        scores = torch.arange(12, 0, -1, dtype=torch.float32)[:, None]
        pos, neg = pairs(*((g, 0) for g in range(6))), pairs(*((g, 0) for g in range(6, 12)))
        self.assertEqual(binary_metrics(scores, pos, neg), {"AUROC": 1.0, "AUPRC": 1.0})
        data = {"phenotype": SimpleNamespace(num_nodes=1)}
        self.assertEqual(split_by_frequency(scores, data, EMPTY, pos, neg),
                         {"tail": {"AUROC": 1.0, "AUPRC": 1.0}})
        self.assertEqual(split_by_frequency(scores, data, EMPTY, pos[:, :4], neg), {})


if __name__ == "__main__":
    unittest.main()
