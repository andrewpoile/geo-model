import numpy as np
from matplotlib import colormaps
from matplotlib.colors import ListedColormap

from geo_model.measures.travel import MILE

AXIS_LABELS = {
    "decile": "Routes from LSOAs at or below IDACI decile",
    "min_distance": "Minimum route distance (m)",
    "capacity_scale": "Route capacity scale (× disadvantaged share of places)",
    "progressivity": "Route seat progressivity (weight f(D)^p)",
    "max_local_p8": "Highest Progress 8 allowed nearby",
    "local_radius": "Local performance radius (m)",
    "performance_weight": "Performance weight, other students",
    "disadvantaged_performance_weight": "Performance weight, disadvantaged students",
    "route_discount": "Route discount, other students",
    "disadvantaged_route_discount": "Route discount, disadvantaged students",
}


COLOURS = {"with routes": "#2a78d6", "without routes": "#52514e"}


# Okabe-Ito hues for the survey modes, and the route in the darkest ink since
# it is the mode the routes add.
MODE_COLOURS = {
    "walk": "#0173b2",
    "cycle": "#029e73",
    "car": "#d55e00",
    "bus": "#de8f05",
    "other": "#cc78bc",
    "route": "#1f2933",
}


# The scenario hues again, the disadvantaged group in full ink and the rest
# washed out, so the unassigned panel reads against the scenario panels.
GROUP_COLOURS = {
    "disadvantaged, with routes": "#2a78d6",
    "disadvantaged, without routes": "#52514e",
    "other, with routes": "#7fb0e3",
    "other, without routes": "#a5a49f",
}


DEFAULT_LINE = {"color": "#898781", "linestyle": "--", "linewidth": 1}


# The stars each value of the index panels takes, by the level its
# Holm-corrected p-value falls below, and the key the figure gives them.
SIGNIFICANCE = [(0.001, "***"), (0.01, "**"), (0.05, "*")]


SIGNIFICANCE_KEY = (
    "Above the index: paired t-test, with routes against without, "
    "Holm-corrected over every cell\n"
    "*** p < 0.001   ** p < 0.01   * p < 0.05   ns p ≥ 0.05   – undefined"
)


# What the intake panels and the FSM strip show of each school: its term of the
# dissimilarity index, diverging either side of the region-wide mix, or the
# disadvantaged share of its intake, in one hue from light to dark.
INTAKE_MEASURES = {
    "dissimilarity": {
        "column": "dissimilarity_term",
        "cmap": "RdBu_r",
        "label": "Share of disadvantaged less share of other students",
    },
    "share": {
        "column": "share",
        "cmap": "Blues",
        "label": "Disadvantaged share of intake",
    },
}


# The map's districts in one hue, a step per IDACI decile and the most deprived
# darkest, stopping short of the hue's darkest so the links stay legible over
# it. Schools diverge either side of the England average Progress 8 of 0.
DECILE_CMAP = ListedColormap(colormaps["Purples"](np.linspace(0.15, 0.75, 10))[::-1])


P8_CMAP = "RdBu"


INK = "#1f2933"


# Two hues neither the fill nor the school scale uses, the green across from
# the purple fill so the plain trips stand out against it.
LINK_COLOURS = {"without a route": "#029e73", "on a route": "#de8f05"}


# The map's scale bar, kilometres ticked above the line and miles below: the
# metres in each unit and the units the bar reaches, near a third of
# Southampton's 10.9km width either way.
SCALE_BAR = {"km": (1000.0, 3), "miles": (MILE, 2)}
