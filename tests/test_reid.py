import numpy as np
import pytest

from reid import (
    IdentityGallery,
    best_detection_index,
    box_iou,
    clamp_box,
    cosine_distance,
    l2_normalize,
)


def unit(*values):
    """Build a unit-length vector from the given components."""
    return l2_normalize(np.asarray(values, dtype=np.float32))


def test_l2_normalize_returns_unit_length():
    assert np.isclose(np.linalg.norm(l2_normalize([3.0, 4.0])), 1.0)


def test_l2_normalize_of_zero_vector_is_still_zero():
    assert not l2_normalize([0.0, 0.0, 0.0]).any()


def test_cosine_distance_of_identical_vectors_is_zero():
    assert cosine_distance(unit(1, 0, 0), unit(1, 0, 0)) == pytest.approx(0.0)


def test_cosine_distance_of_orthogonal_vectors_is_one():
    assert cosine_distance(unit(1, 0, 0), unit(0, 1, 0)) == pytest.approx(1.0)


def test_cosine_distance_of_opposite_vectors_is_two():
    assert cosine_distance(unit(1, 0, 0), unit(-1, 0, 0)) == pytest.approx(2.0)


def test_cosine_distance_ignores_magnitude():
    assert cosine_distance([5.0, 0.0], [0.5, 0.0]) == pytest.approx(0.0)


def test_match_returns_the_nearest_candidate_not_the_first_one():
    gallery = IdentityGallery(match_threshold=0.6, match_margin=0.05)
    near = gallery.create(unit(1.0, 0.0, 0.0, 0.0))
    gallery.create(unit(0.0, 1.0, 0.0, 0.0))

    identity, _ = gallery.match(unit(0.95, 0.05, 0.0, 0.0))

    assert identity == near


def test_match_refuses_when_two_candidates_are_equally_close():
    gallery = IdentityGallery(match_threshold=0.6, match_margin=0.05)
    gallery.create(unit(1.0, 0.1, 0.0, 0.0))
    gallery.create(unit(1.0, -0.1, 0.0, 0.0))

    identity, _ = gallery.match(unit(1.0, 0.0, 0.0, 0.0))

    assert identity is None


def test_match_refuses_candidates_beyond_the_threshold():
    gallery = IdentityGallery(match_threshold=0.3)
    gallery.create(unit(1.0, 0.0, 0.0, 0.0))

    identity, distance = gallery.match(unit(0.0, 1.0, 0.0, 0.0))

    assert identity is None
    assert distance == pytest.approx(1.0)


def test_match_skips_identities_that_are_still_claimed():
    gallery = IdentityGallery(match_threshold=0.6, match_margin=0.05)
    busy = gallery.create(unit(1.0, 0.0, 0.0, 0.0))
    gallery.create(unit(0.0, 1.0, 0.0, 0.0))

    identity, _ = gallery.match(unit(1.0, 0.0, 0.0, 0.0), exclude={busy})

    assert identity is None


def test_match_returns_none_for_an_empty_gallery():
    identity, distance = IdentityGallery().match(unit(1.0, 0.0))

    assert identity is None
    assert distance == pytest.approx(2.0)


def test_match_returns_none_for_a_degenerate_query():
    gallery = IdentityGallery()
    gallery.create(unit(1.0, 0.0))

    assert gallery.match([0.0, 0.0])[0] is None


def test_identities_are_never_reused():
    gallery = IdentityGallery()
    first = gallery.create(unit(1.0, 0.0))
    second = gallery.create(unit(0.0, 1.0))

    assert first != second
    assert gallery.ids == {first, second}


def test_observe_moves_the_appearance_model_only_gradually():
    gallery = IdentityGallery(ema_alpha=0.9)
    identity = gallery.create(unit(1.0, 0.0, 0.0))
    before = gallery.match(unit(1.0, 0.0, 0.0))[1]

    gallery.observe(identity, unit(0.0, 1.0, 0.0))

    after = gallery.match(unit(1.0, 0.0, 0.0))[1]
    assert after > before
    assert after < 0.1


def test_observe_ignores_a_degenerate_observation():
    gallery = IdentityGallery()
    identity = gallery.create(unit(1.0, 0.0))
    before = gallery.match(unit(1.0, 0.0))[1]

    gallery.observe(identity, [0.0, 0.0])

    assert gallery.match(unit(1.0, 0.0))[1] == pytest.approx(before)


def test_end_frame_forgets_identities_that_stayed_unclaimed():
    gallery = IdentityGallery(max_age=2)
    gone = gallery.create(unit(1.0, 0.0))
    kept = gallery.create(unit(0.0, 1.0))

    for _ in range(3):
        gallery.end_frame(claimed={kept})

    assert gallery.ids == {kept}
    assert gone not in gallery.ids


def test_end_frame_resets_the_age_of_claimed_identities():
    gallery = IdentityGallery(max_age=1)
    identity = gallery.create(unit(1.0, 0.0))

    for _ in range(5):
        gallery.end_frame(claimed={identity})

    assert identity in gallery.ids


def test_box_iou_of_identical_boxes_is_one():
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)


def test_box_iou_of_disjoint_boxes_is_zero():
    assert box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == pytest.approx(0.0)


def test_box_iou_of_half_overlap():
    assert box_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1.0 / 3.0)


def test_box_iou_of_degenerate_boxes_is_zero():
    assert box_iou([0, 0, 0, 0], [0, 0, 0, 0]) == pytest.approx(0.0)


def test_best_detection_index_finds_the_overlapping_detection():
    boxes = [[0, 0, 10, 10], [100, 100, 140, 200]]

    assert best_detection_index([102, 104, 138, 196], boxes) == 1


def test_best_detection_index_ignores_list_order():
    boxes = [[100, 100, 140, 200], [0, 0, 10, 10], [50, 50, 70, 90]]

    assert best_detection_index([0, 0, 10, 10], boxes) == 1


def test_best_detection_index_is_none_when_nothing_overlaps():
    boxes = [[0, 0, 10, 10], [100, 100, 140, 200]]

    assert best_detection_index([300, 300, 340, 400], boxes) is None


def test_best_detection_index_is_none_for_an_empty_frame():
    assert best_detection_index([0, 0, 10, 10], []) is None


def test_best_detection_index_picks_the_closer_of_two_overlaps():
    boxes = [[0, 0, 10, 10], [1, 0, 11, 10]]

    assert best_detection_index([0, 0, 10, 10], boxes) == 0
def test_clamp_box_keeps_boxes_inside_the_frame():
    assert clamp_box(-5, -3, 50, 60, 100, 80) == (0, 0, 50, 60)


def test_clamp_box_trims_boxes_that_overflow_the_frame():
    assert clamp_box(90, 70, 200, 300, 100, 80) == (90, 70, 100, 80)


def test_clamp_box_always_returns_a_non_empty_crop():
    left, top, right, bottom = clamp_box(100, 80, 100, 80, 100, 80)

    assert right > left
    assert bottom > top


def test_clamp_box_keeps_a_one_pixel_box_usable():
    left, top, right, bottom = clamp_box(10, 10, 10.4, 10.4, 100, 80)

    assert (left, top) == (10, 10)
    assert right >= 11
    assert bottom >= 11
