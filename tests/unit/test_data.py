import numpy as np
from sit_lmpc.data.normalization import Normalization
from sit_lmpc.data.laps import Lap, load_laps, save_laps


def test_constant_features_and_targets_are_finite():
    data = np.ones((12, 6))
    normalizer = Normalization.fit(data)
    np.testing.assert_allclose(normalizer.encode_inputs(data[:, :5]), 0)
    np.testing.assert_allclose(normalizer.encode_targets(data[:, -1:]), 0)
    np.testing.assert_allclose(normalizer.decode_targets(np.zeros((12, 1))), 1)


def test_lap_roundtrip_without_pickle(tmp_path):
    lap = Lap(np.ones((10, 7)), np.ones((10, 7)), np.ones((10, 2)), np.arange(1, 11) * 0.1)
    save_laps(tmp_path / "laps.npz", [lap])
    restored = load_laps(tmp_path / "laps.npz")[0]
    np.testing.assert_array_equal(restored.cartesian, lap.cartesian)
    np.testing.assert_array_equal(restored.controls, lap.controls)
