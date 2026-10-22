import pandas as pd
import torch
import unittest

from crossvul_dg.data import ClassBalancedBatchSampler, validate_lopo_frames
from crossvul_dg.losses import CCDROState


def _frame(project, labels):
    return pd.DataFrame({
        "func": [f"int {project}_{i}(){{return {i};}}" for i in range(len(labels))],
        "label": labels,
        "project": [project] * len(labels),
        "commit_date": pd.date_range("2020-01-01", periods=len(labels), tz="UTC"),
    })


class TrainingInvariantTests(unittest.TestCase):
    def test_balanced_sampler_puts_both_classes_in_every_batch(self):
        sampler = ClassBalancedBatchSampler([0] * 19 + [1], batch_size=4, seed=3)
        labels = [0] * 19 + [1]
        for indices in sampler:
            self.assertEqual({labels[i] for i in indices}, {0, 1})

    def test_ccdro_present_weights_are_renormalized(self):
        state = CCDROState()
        state.q = {(0, 0): torch.tensor(0.1), (0, 1): torch.tensor(0.2),
                   (1, 0): torch.tensor(0.3), (1, 1): torch.tensor(0.4)}
        value = state.weighted_loss({(0, 0): torch.tensor(2.0),
                                     (0, 1): torch.tensor(4.0)})
        self.assertAlmostEqual(value.item(), (0.1 * 2 + 0.2 * 4) / 0.3, places=5)

    def test_fold_validation_rejects_single_class_test(self):
        train = pd.concat([_frame("a", [0, 1]), _frame("b", [0, 1])])
        with self.assertRaisesRegex(ValueError, "test split"):
            validate_lopo_frames(train, _frame("v", [0, 1]), _frame("t", [0, 0]))

    def test_fold_validation_requires_source_cells_for_ccdro(self):
        train = pd.concat([_frame("a", [0, 1]), _frame("b", [0, 0])])
        with self.assertRaisesRegex(ValueError, "source project"):
            validate_lopo_frames(train, _frame("v", [0, 1]), _frame("t", [0, 1]),
                                 require_source_cells=True)


if __name__ == "__main__":
    unittest.main()
