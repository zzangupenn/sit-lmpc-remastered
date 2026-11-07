import numpy as np
import pytest
from sit_lmpc.track import Track, map_path


@pytest.mark.parametrize("name", ["custom1", "Spielberg"])
def test_track_coordinate_roundtrip(name):
    track = Track.from_track_path(map_path(name))
    for fraction in (0.05, 0.25, 0.6):
        source = np.array([float(track.s_frame_max) * fraction, 0.1, 0.05])
        cartesian = np.asarray(track.frenet_to_cartesian(*source))
        restored = np.asarray(track.cartesian_to_frenet(*cartesian, s_guess=source[0]))
        assert np.isfinite(restored).all()
        np.testing.assert_allclose(restored, source, atol=0.1, rtol=0)


def test_unknown_map_does_not_download():
    with pytest.raises(FileNotFoundError):
        map_path("nonexistent-track")
