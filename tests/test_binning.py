"""Equal-width and equal-mass binning.

Boundary assignment is where binning bugs live: a confidence of exactly 1.0
must not overflow the last bin, and equal-mass chunking must stay
deterministic and lossless.
"""

import pytest

from miscal import EmptyDatasetError, MiscalError, equal_mass_bins, equal_width_bins, make_bins


def test_equal_width_edges_partition_the_unit_interval():
    bins = equal_width_bins([0.5], [1], n_bins=4)
    assert [(b.lower, b.upper) for b in bins] == [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]


def test_boundary_assignment_is_lower_inclusive_and_one_lands_in_the_last_bin():
    # 0.5 sits exactly on the boundary between [0.4,0.5) and [0.5,0.6).
    bins = equal_width_bins([0.5], [1], n_bins=10)
    assert bins[5].count == 1
    assert bins[4].count == 0
    # Exactly 1.0 must not overflow past the final bin.
    bins = equal_width_bins([1.0], [1], n_bins=10)
    assert bins[-1].count == 1
    assert sum(b.count for b in bins) == 1


def test_counts_sum_to_dataset_size_and_empty_bins_are_zeroed():
    confs = [0.05, 0.55, 0.95, 0.95]
    bins = equal_width_bins(confs, [0, 1, 1, 0], n_bins=10)
    assert sum(b.count for b in bins) == 4
    empty = [b for b in bins if b.count == 0]
    assert len(empty) == 7
    assert all(b.mean_confidence == 0.0 and b.accuracy == 0.0 for b in empty)
    # Occupied bins report within-bin means, not bin midpoints.
    top = equal_width_bins([0.91, 0.99], [1, 0], n_bins=10)[-1]
    assert top.mean_confidence == pytest.approx(0.95)
    assert top.accuracy == pytest.approx(0.5)


def test_gap_is_signed_confidence_minus_accuracy():
    bins = equal_width_bins([0.9, 0.9], [0, 0], n_bins=10)
    assert bins[-1].gap == pytest.approx(0.9)  # overconfident
    bins = equal_width_bins([0.1, 0.1], [1, 1], n_bins=10)
    assert bins[1].gap == pytest.approx(-0.9)  # underconfident


def test_equal_mass_bins_have_near_equal_sizes_and_data_driven_edges():
    confs = [i / 100 for i in range(10)]
    bins = equal_mass_bins(confs, [1] * 10, n_bins=3)
    assert [b.count for b in bins] == [4, 3, 3]  # remainder spread over the first bins
    assert sum(b.count for b in bins) == 10
    # Edges are the actual min/max confidence inside each chunk.
    bins = equal_mass_bins([0.1, 0.2, 0.3, 0.7, 0.8, 0.9], [1] * 6, n_bins=2)
    assert (bins[0].lower, bins[0].upper) == (0.1, 0.3)
    assert (bins[1].lower, bins[1].upper) == (0.7, 0.9)


def test_equal_mass_with_fewer_records_than_bins_never_creates_empty_bins():
    bins = equal_mass_bins([0.2, 0.8], [1, 0], n_bins=10)
    assert len(bins) == 2
    assert all(b.count == 1 for b in bins)


def test_equal_mass_sorting_is_deterministic_under_ties():
    confs = [0.5] * 6
    a = equal_mass_bins(confs, [1, 0, 1, 0, 1, 0], n_bins=2)
    b = equal_mass_bins(confs, [1, 0, 1, 0, 1, 0], n_bins=2)
    assert a == b


def test_make_bins_dispatches_by_strategy():
    confs, outs = [0.1, 0.9], [0, 1]
    assert make_bins(confs, outs, 4, "width") == equal_width_bins(confs, outs, 4)
    assert make_bins(confs, outs, 4, "mass") == equal_mass_bins(confs, outs, 4)


def test_invalid_inputs_raise_descriptive_errors():
    with pytest.raises(MiscalError, match="unknown binning strategy"):
        make_bins([0.5], [1], 4, "kmeans")
    with pytest.raises(MiscalError, match="differ in length"):
        equal_width_bins([0.5, 0.6], [1], n_bins=4)
    with pytest.raises(EmptyDatasetError):
        equal_width_bins([], [], n_bins=4)
    with pytest.raises(MiscalError, match="n_bins"):
        equal_width_bins([0.5], [1], n_bins=0)


def test_out_of_range_confidences_are_rejected_not_misbinned():
    # A negative confidence would index buckets[-1] and land in the *top*
    # width bin — the worst possible silent corruption for a calibration tool.
    for bad in (-0.15, 1.5, float("nan")):
        with pytest.raises(MiscalError, match="outside"):
            equal_width_bins([0.5, bad], [1, 0], n_bins=10)
        with pytest.raises(MiscalError, match="outside"):
            equal_mass_bins([0.5, bad], [1, 0], n_bins=10)
