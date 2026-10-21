import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from shapely import box

from geo_model.data import sources
from geo_model.plots.maps import draw_map, plot_map
from geo_model.plots.style import SCALE_BAR

# --------------------------------------------------------------------------
# draw_map and plot_map
# --------------------------------------------------------------------------


def synthetic_city():
    """Two 10km districts, two schools and four students, matched both ways.

    With routes, student 0 is seated at school 0, student 1 at school 1 on
    route 0 and student 2 at school 1, while student 3 is left unassigned.
    Without routes, student 1 is seated at school 0 instead.
    """
    areas = gpd.GeoDataFrame(
        {"IDACI Decile": [1, 10]},
        geometry=[box(0, 0, 10_000, 10_000), box(10_000, 0, 20_000, 10_000)],
        crs=sources.CRS,
    )
    schools = pd.DataFrame(
        {
            "Easting": [5_000.0, 15_000.0],
            "Northing": [5_000.0, 5_000.0],
            "P8MEA": [-0.5, 0.25],
        }
    )
    student_xy = np.array(
        [
            [1_000.0, 1_000.0],
            [2_000.0, 8_000.0],
            [12_000.0, 3_000.0],
            [18_000.0, 8_000.0],
        ]
    )
    matchings = {
        "with routes": np.array([[0, -1], [1, 0], [1, -1], [-1, -1]]),
        "without routes": np.array([[0, -1], [0, -1], [1, -1], [-1, -1]]),
    }
    return areas, schools, student_xy, matchings


def test_draw_map_links_every_seated_student_to_their_school():
    areas, schools, student_xy, matchings = synthetic_city()
    fig = Figure(figsize=(16, 8), layout="constrained")

    draw_map(fig, areas, schools, student_xy, matchings)

    school_xy = schools[["Easting", "Northing"]].to_numpy()
    linked = {
        "with routes": {"without a route": [0, 2], "on a route": [1]},
        "without routes": {"without a route": [0, 1, 2], "on a route": []},
    }
    # The panels come first and in scenario order; the colorbars follow them.
    for ax, (scenario, matching) in zip(fig.axes, matchings.items()):
        assert ax.get_title() == scenario
        drawn = {collection.get_label(): collection for collection in ax.collections}
        deciles = drawn["districts"].get_array()
        assert deciles is not None and list(deciles) == [1, 10]
        assert drawn["districts"].get_clim() == (0.5, 10.5)
        for label, students in linked[scenario].items():
            links = drawn[label]
            assert isinstance(links, LineCollection)
            assert [s.tolist() for s in links.get_segments()] == [
                [student_xy[i].tolist(), school_xy[matching[i, 0]].tolist()]
                for i in students
            ]
        np.testing.assert_array_equal(drawn["seated"].get_offsets(), student_xy[:3])
        np.testing.assert_array_equal(drawn["unassigned"].get_offsets(), student_xy[3:])
        scores = drawn["schools"].get_array()
        assert scores is not None and list(scores) == [-0.5, 0.25]
        # Centred on the England average, reaching the largest score either way.
        assert drawn["schools"].get_clim() == (-0.5, 0.5)
        # A metre is as long north as east, so the scale holds in every direction.
        assert ax.get_aspect() == 1.0
        # Kilometres ticked above the line and miles below, from one origin.
        origins = []
        for (unit, (metres, length)), side in zip(SCALE_BAR.items(), (1, -1)):
            scale = drawn[unit]
            assert isinstance(scale, LineCollection)
            (start, end), *ticks = scale.get_segments()
            assert start[1] == end[1]
            assert end[0] - start[0] == pytest.approx(length * metres)
            assert [tick[0, 0] - start[0] for tick in ticks] == pytest.approx(
                [i * metres for i in range(length + 1)]
            )
            assert all(np.sign(tick[1, 1] - tick[0, 1]) == side for tick in ticks)
            origins.append(start.tolist())
        assert origins[0] == origins[1]
        labels = ["0", "1", "2", "3 km", "0", "1", "2 miles"]
        assert [text.get_text() for text in ax.texts] == labels


def test_plot_map_writes_a_png(tmp_path):
    png = tmp_path / "out" / "map.png"

    plot_map(*synthetic_city(), png)

    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
