import unittest

import numpy as np

from stem_projects.stock import chronological_split, classification_metrics, make_examples


class StockTests(unittest.TestCase):
    def test_features_precede_next_day_labels(self):
        prices = np.array([100, 110, 110, 99, 108], dtype=float)
        x, y = make_examples(prices)
        np.testing.assert_allclose(x[:, 0], np.log(prices[1:-1] / prices[:-2]))
        np.testing.assert_array_equal(y, [1, 0, 2])  # flat, down, up

    def test_chronological_holdout_does_not_shuffle(self):
        x, y = np.arange(10).reshape(10, 1), np.arange(10)
        train_x, test_x, train_y, test_y = chronological_split(x, y, .7)
        np.testing.assert_array_equal(train_x[:, 0], np.arange(7))
        np.testing.assert_array_equal(test_x[:, 0], np.arange(7, 10))
        np.testing.assert_array_equal(train_y, np.arange(7))
        np.testing.assert_array_equal(test_y, np.arange(7, 10))

    def test_confusion_matrix_rows_are_actual_labels(self):
        report = classification_metrics([0, 1, 2, 2], [0, 2, 0, 2])
        self.assertEqual(report["accuracy"], .5)
        self.assertEqual(report["confusion_matrix_rows_actual"],
                         [[1, 0, 0], [0, 0, 1], [1, 0, 1]])


if __name__ == "__main__":
    unittest.main()

