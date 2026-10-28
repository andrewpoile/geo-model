import math
from dataclasses import dataclass, field

SEED = 20260907
# Samples drawn unless --seeds says otherwise by run.py and sweep.py, and by
# optimise.py for the compromise it runs on held-out samples. optimise.py's
# search and the scenarios set counts of their own.
N_SEEDS = 30

# Routes are built only from LSOAs at or below this IDACI decile, 1 being the
# most deprived 10% of LSOAs. Under DISADVANTAGE "decile" it also sets which
# students are disadvantaged.
DISADVANTAGED_DECILE = 3

# Who is disadvantaged, and who may ride a route leaving their district:
#   "score"       Each LSOA's students are drawn disadvantaged in proportion to
#                 its IDACI score, and only they ride.
#   "score-open"  Drawn the same way, but every student of a route's district
#                 rides, as the SCT model's district-level routes.
#   "decile"      Every student of an LSOA at or below the decile is
#                 disadvantaged and rides: the model as it was before the
#                 scores were drawn from.
# Under every choice route seats are sized on the disadvantaged students.
DISADVANTAGE_CHOICES = ("score", "score-open", "decile")
DISADVANTAGE = "score"

# Metres. A route carries a student to a school they cannot reach unaided, so
# nearer schools need none. 3218m is two miles, the lower bound of the extended
# rights to free home-to-school travel held by low-income 11-16 year olds.
MIN_ROUTE_DISTANCE = 3218

# Minutes. Districts are routed only to schools a bus reaches within this at
# the weekday morning peak, along the OS Multi-modal Routing Network. No cap
# by default, so the route set is the distance condition's alone.
MAX_ROUTE_MINUTES = math.inf

# Seats on the routes into a school, as a multiple of the disadvantaged
# students' fair share of the Year 7 places it offers: k * places * D / N, with
# D the region's disadvantaged students and N its Year 7 cohort, both on the
# mean cohort over the years, so they do not change with a sample. At 1 a
# school's routes hold the seats its disadvantaged students would take if
# every intake matched the region's mix. The seats are split between the
# school's routes in proportion to the disadvantaged students of each route's
# district. Route capacities are exogenous to the SCT instance.
ROUTE_CAPACITY_SCALE = 1.0

# How far a school's route seats lean towards the more deprived districts,
# p >= 0. A district at IDACI decile D at or below the threshold T is weighted
# f(D)^p, f running from 1 at decile 1 to 1 / T at decile T, so 0 splits a
# school's seats on its routes' disadvantaged students alone and p weights
# decile 1 T^p times decile T. The weights only split each school's seats
# between its routes, so they move seats between districts without changing
# how many the school holds.
ROUTE_PROGRESSIVITY = 1.0

# The profile f of the progressive weight: 1 / D, or with True the linear
# (T + 1 - D) / T. The two agree at deciles 1 and T and differ between them.
LINEAR_PROGRESSIVITY = False

# A school's route seats are rounded to the nearest whole seat, then split
# between its routes by largest remainder, so they sum to it exactly, and a
# route apportioned none is not built. True rounds each school's seats up
# instead, so a school with a route keeps at least one seat on it.
ROUND_UP_SEATS = False

# True keeps, for each school, only its route to the nearest district holding
# a seat on one, so every school offers a single route.
SHORTEST_ROUTE_ONLY = False

# True routes from only as many districts as there are schools, whatever their
# decile. The route-eligible districts are taken in IDACI rank order, the most
# deprived in England first, each kept if it and those kept before it can still
# each have a school of its own beyond the minimum distance, so a district that
# cannot is passed over for the next. Each school is paired with one of them so
# the longest route is as short as it can be, ties going to the least total
# length. Every school offers a single route.
BOTTLENECK_ROUTES = False

# True routes from only as many districts as there are schools, those holding
# the most disadvantaged students, whatever their decile, a tie going to the
# lower IDACI rank. Taking them in that order, each district is paired with the
# school offering the most places that no district before it took, a tie going
# to the nearer. Every school offers a single route.
LARGEST_FIRST_ROUTES = False

# A route carries a student past their local schools, so a district already
# within reach of a school performing above this needs none. P8MEA is centred
# on the England average, so 0 is an average school.
MAX_LOCAL_P8 = 0.0

# Metres. 1609m is one mile, read the other way round from MIN_ROUTE_DISTANCE:
# the distance a district is taken to be served over rather than the distance
# it must be carried. The two are independent, so they are swept apart.
LOCAL_RADIUS = 1609

# Share of the secondary preference ranking driven by Progress 8 rather than
# distance, for students who are not disadvantaged. Distances and scores are
# each scaled by their own spread over the region, so the trade-off is the
# region's: in Southampton, at 0.3, one Progress 8 point is worth roughly
# 1.4km of extra travel.
PERFORMANCE_WEIGHT = 0.3

# Share of the travel a route takes out of the ranking of the school it serves,
# for students who are not disadvantaged. At 0.5 a routed school ranks
# as if it stood half as far away, so a route is worth taking but does not put
# every distant school ahead of the local one.
ROUTE_DISCOUNT = 0.5

# The same two shares for disadvantaged students. They default
# to the other students' own, so the population is homogeneous until a sweep
# moves one group away from the other.
DISADVANTAGED_PERFORMANCE_WEIGHT = PERFORMANCE_WEIGHT
DISADVANTAGED_ROUTE_DISCOUNT = ROUTE_DISCOUNT

# How a school orders the bundles proposed to it:
#   "distance"  By the student's distance alone, a routed bundle's discounted
#               by PRIORITY_DISCOUNT, a tie going to the nearer on the true
#               distance.
#   "bracket"   Every routed bundle and every student of the school's own
#               district first, everyone else after, each bracket by
#               distance: the model as it was before.
PRIORITY_CHOICES = ("distance", "bracket")
PRIORITY = "distance"

# Share of a routed student's distance a school's priority takes off under
# PRIORITY "distance". At 0.5 a routed bundle ranks as if the student lived
# half as far away.
PRIORITY_DISCOUNT = 0.5

# Standard deviation of every student's taste for each school, added to its
# preference cost, in the cost's units: those of a standard deviation of
# distance. Drawn from a seed of the sample's own, so a sample ranks with the
# same tastes with routes and without, and at every setting.
PREFERENCE_NOISE = 0.5


@dataclass(frozen=True)
class Settings:
    """The parameters a matching can be scored under, at the model's defaults,
    each field's metadata holding the help of its command-line option."""

    decile: int = field(
        default=DISADVANTAGED_DECILE,
        metadata={
            "help": "routes are built only from LSOAs at or below this IDACI decile"
        },
    )
    min_distance: float = field(
        default=MIN_ROUTE_DISTANCE,
        metadata={
            "help": (
                "districts are routed only to schools further away than this, in metres"
            )
        },
    )
    max_route_minutes: float = field(
        default=MAX_ROUTE_MINUTES,
        metadata={
            "help": (
                "districts are routed only to schools a bus reaches within this many "
                "minutes at the weekday morning peak"
            )
        },
    )
    capacity_scale: float = field(
        default=ROUTE_CAPACITY_SCALE,
        metadata={
            "help": (
                "seats on a school's route as a multiple of the disadvantaged students'"
                " fair share of the places it offers"
            )
        },
    )
    progressivity: float = field(
        default=ROUTE_PROGRESSIVITY,
        metadata={
            "help": (
                "how far a school's route seats lean towards the more deprived "
                "districts, 0 flat"
            )
        },
    )
    max_local_p8: float = field(
        default=MAX_LOCAL_P8,
        metadata={
            "help": (
                "a district with a school scoring above this Progress 8 within the "
                "local radius is not routed"
            )
        },
    )
    local_radius: float = field(
        default=LOCAL_RADIUS,
        metadata={
            "help": (
                "metres around a district's centroid its local schools are read from"
            )
        },
    )
    performance_weight: float = field(
        default=PERFORMANCE_WEIGHT,
        metadata={
            "help": (
                "share of the preference ranking driven by Progress 8 rather than "
                "travel, for students who are not disadvantaged"
            )
        },
    )
    route_discount: float = field(
        default=ROUTE_DISCOUNT,
        metadata={
            "help": (
                "share of the travel a route takes out of the ranking of the school it "
                "serves, for students who are not disadvantaged"
            )
        },
    )
    disadvantaged_performance_weight: float = field(
        default=DISADVANTAGED_PERFORMANCE_WEIGHT,
        metadata={"help": "as --performance-weight, for disadvantaged students"},
    )
    disadvantaged_route_discount: float = field(
        default=DISADVANTAGED_ROUTE_DISCOUNT,
        metadata={"help": "as --route-discount, for disadvantaged students"},
    )
    priority_discount: float = field(
        default=PRIORITY_DISCOUNT,
        metadata={
            "help": (
                "share of a routed student's distance a school's priority takes off, "
                "under --priority distance"
            )
        },
    )
    shortest_only: bool = field(
        default=SHORTEST_ROUTE_ONLY,
        metadata={
            "help": (
                "keep only each school's route to its nearest route-eligible district"
            )
        },
    )
    bottleneck: bool = field(
        default=BOTTLENECK_ROUTES,
        metadata={
            "help": (
                "route only as many districts as there are schools, the most deprived "
                "by IDACI rank that can each have a school of their own, each school "
                "paired with one so the longest route is as short as it can be"
            )
        },
    )
    largest_first: bool = field(
        default=LARGEST_FIRST_ROUTES,
        metadata={
            "help": (
                "route only the districts holding the most disadvantaged students, as "
                "many as there are schools, each paired in turn with the free school "
                "offering the most places"
            )
        },
    )


DEFAULTS = Settings()

# The two scenarios every sample is matched in: with the routes, and the same
# students with none, the baseline the routes are measured against.
SCENARIOS = ("with routes", "without routes")
