"""Checks for emission_line_fitter.adopted_broad_sigma: the fixed broad width isn't taken from fits pinned at a bound."""
import numpy as np

from emission_line_fitter import adopted_broad_sigma

BOUNDS = (2.0, 20.0)


def test_interior_median_is_kept_even_with_some_fits_at_the_bound():
    # 312D's situation: a third of the fits are pinned, but the median is a real value.
    assert adopted_broad_sigma([10.0, 12.0, 18.0, 20.0, 20.0], BOUNDS) == 18.0


def test_median_on_the_upper_bound_falls_back_to_the_off_bound_fits():
    # 127I's situation: over half the fits are pinned at 20 A.
    assert adopted_broad_sigma([10.0, 12.0, 14.0, 20.0, 20.0, 20.0, 20.0], BOUNDS) == 12.0


def test_median_on_the_lower_bound_falls_back_too():
    assert adopted_broad_sigma([2.0, 2.0, 2.0, 8.0, 10.0], BOUNDS) == 9.0


def test_all_fits_pinned_returns_the_bound_rather_than_failing():
    assert adopted_broad_sigma(np.full(5, 20.0), BOUNDS) == 20.0
