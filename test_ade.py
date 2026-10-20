"""Checks for Equation (11) as printed and offline forecast alignment."""

import contextlib
import io
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ade import ade, ade_derivative, ade_forecast


WINDOWS = [10, 20, 40, 65, 100]


class ADETests(unittest.TestCase):
    def test_constant_price_reproduces_the_printed_equations_bias(self):
        for window in WINDOWS:
            for price in [0.0, 100.0, 1000.0]:
                with self.subTest(window=window, price=price):
                    y = pd.Series([price] * 160)
                    # Document the printed equation's behavior, not a true
                    # constant-signal derivative (which would be zero).
                    expected = -12 * price / window**2
                    self.assertAlmostEqual(ade_derivative(y, 120, window), expected)
                    self.assertAlmostEqual(ade_forecast(y, 120, window, 5), price + 5 * expected)

    def test_window_uses_previous_prices_and_excludes_current_and_future_rows(self):
        y = pd.Series(np.zeros(160))
        window, k = 20, 100
        changed = y.copy()
        changed.iloc[:k - window - 1] = 1000
        changed.iloc[k:] = 1000
        self.assertEqual(ade_derivative(changed, k, window), 0.0)
        changed.iloc[k - 1] = 1
        self.assertGreater(ade_derivative(changed, k, window), 0.0)
        changed.iloc[k - 1] = 0
        changed.iloc[k - window - 1] = 1
        self.assertLess(ade_derivative(changed, k, window), 0.0)

    def test_window_is_available_at_row_n_plus_one(self):
        y = pd.Series([100.0] * 102)
        self.assertEqual(ade_derivative(y, 100, 100), 0.0)
        self.assertAlmostEqual(ade_derivative(y, 101, 100), -.12)
        with self.assertRaises(ValueError):
            ade_derivative(y, 10, 0)
        with self.assertRaises(IndexError):
            ade_derivative(y, len(y), 100)

    def test_comparison_preserves_target_dates_and_shared_plot_dates(self):
        prices = pd.DataFrame(
            {'Close': 100 + np.arange(180) * .1 + np.sin(np.arange(180))},
            index=pd.bdate_range('2025-01-01', periods=180)
        )
        try:
            with patch('ade.init', return_value=prices.copy()), \
                    patch('ade.plt.show') as show, \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                result = ade()
            self.assertEqual(show.call_count, 1)
            for window in WINDOWS:
                prediction = result[f'ade_pred_{window}']
                self.assertEqual(prediction.first_valid_index(), prices.index[window + 6])
                self.assertAlmostEqual(prediction.iloc[window + 6],
                                       ade_forecast(prices['Close'], window + 1, window, 5))
                pd.testing.assert_series_equal(result[f'ade_error_{window}'],
                                               prediction - result['Close'], check_names=False)
                self.assertIn(f'Errors for N_ADE={window}', output.getvalue())
            axes = plt.gcf().axes
            self.assertEqual(len(axes), 2)
            self.assertEqual(len(axes[0].lines), 6)
            for line in axes[0].lines:
                self.assertEqual(pd.Timestamp(line.get_xdata()[0]), prices.index[106]) # type: ignore
        finally:
            plt.close('all')


if __name__ == '__main__':
    unittest.main()
