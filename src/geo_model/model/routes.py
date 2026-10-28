import bisect

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from geo_model.model.settings import (
    BOTTLENECK_ROUTES,
    DISADVANTAGE,
    DISADVANTAGE_CHOICES,
    DISADVANTAGED_DECILE,
    LARGEST_FIRST_ROUTES,
    LINEAR_PROGRESSIVITY,
    LOCAL_RADIUS,
    MAX_LOCAL_P8,
    MAX_ROUTE_MINUTES,
    MIN_ROUTE_DISTANCE,
    ROUND_UP_SEATS,
    ROUTE_CAPACITY_SCALE,
    ROUTE_PROGRESSIVITY,
    SHORTEST_ROUTE_ONLY,
)
from geo_model.model.students import disadvantaged_cohort


class EmptyRouteSet(ValueError):
    """No route survives the conditions: an outcome of the settings, not bad input."""


def centroid_xy(districts: gpd.GeoDataFrame) -> np.ndarray:
    """Read the population centroid of every district as coordinates.

    Args:
        districts (gpd.GeoDataFrame): Districts carrying a "Centroids" geometry.

    Returns:
        np.ndarray: Coordinates of shape (n_districts, 2), in the CRS of the
        centroids.
    """
    centroids = np.asarray(districts["Centroids"])
    return np.column_stack((shapely.get_x(centroids), shapely.get_y(centroids)))


def apportion(quotas: np.ndarray, totals: np.ndarray) -> np.ndarray:
    """Round each column of `quotas` to whole seats summing to its total.

    By largest remainder: every entry takes the floor of its quota, and the
    seats a column still lacks go one each to its largest remainders, the
    first row winning a tie.

    Args:
        quotas (np.ndarray): Non-negative quotas, shape (n_rows, n_columns),
        each column summing to its total.

        totals (np.ndarray): Whole seats in each column, shape (n_columns,).

    Returns:
        np.ndarray: Whole seats, shape (n_rows, n_columns), each column
        summing to its total.
    """
    seats = np.floor(quotas)
    order = np.argsort(seats - quotas, axis=0, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(len(quotas))[:, None], axis=0)
    return (seats + (rank < totals - seats.sum(axis=0))).astype(np.int32)


def matching_size(allowed: np.ndarray) -> int:
    """The most districts that can each have a school of their own.

    Args:
        allowed (np.ndarray): Whether each district may be routed to each
        school, shape (n_districts, n_schools).

    Returns:
        int: Pairs in a maximum matching of the districts to the schools.
    """
    # The district matched to each school, -1 for none.
    matched = maximum_bipartite_matching(csr_matrix(allowed), perm_type="row")
    return int((matched >= 0).sum())


def bottleneck_pairs(distances: np.ndarray) -> np.ndarray:
    """Pair every school with a district of its own so the longest pair is as
    short as it can be.

    The longest pair is the least length L at which the pairs no longer than L
    still give every school a district of its own. Of the pairings within L,
    the one of least total length is taken, so the pairs short of the longest
    are as short as they can be too.

    Args:
        distances (np.ndarray): Distance from every district to every school,
        shape (n_districts, n_schools), np.inf where a pair is barred.

    Returns:
        np.ndarray: Whether each pair is chosen, shape (n_districts,
        n_schools), one pair in every column and at most one in every row.

    Raises:
        ValueError: If no pairing gives every school a district of its own.
    """

    def complete(longest: float) -> bool:
        return matching_size(distances <= longest) == distances.shape[1]

    lengths = np.unique(distances[np.isfinite(distances)])
    # A pairing within a length is one within every greater length too, so
    # `complete` is False below L and True from it on.
    at = bisect.bisect_left(lengths, True, key=complete)
    if at == len(lengths):
        raise ValueError(
            f"No pairing gives each of the {distances.shape[1]} schools a "
            f"district of its own among the {distances.shape[0]} they may be "
            "routed to."
        )
    districts, schools = linear_sum_assignment(
        np.where(distances <= lengths[at], distances, np.inf)
    )
    pairs = np.zeros(distances.shape, dtype=bool)
    pairs[districts, schools] = True
    return pairs


def first_districts(allowed: np.ndarray) -> np.ndarray:
    """Take the districts in order, keeping each that can still have a school
    of its own alongside those kept before it, until every school has one.

    A district is passed over when it may be routed to no school, or only to
    schools the districts kept before it need. The sets of districts that can
    each have a school of their own form a transversal matroid, so taking them
    greedily in order keeps the earliest set there is: its k-th district comes
    no later than the k-th of any other set giving every school a district.

    Args:
        allowed (np.ndarray): Whether each district may be routed to each
        school, shape (n_districts, n_schools), the districts in the order
        they are taken.

    Returns:
        np.ndarray: Whether each district is kept, shape (n_districts,), as
        many kept as there are schools.

    Raises:
        EmptyRouteSet: If no set of the districts gives every school a
        district of its own.
    """
    n_schools = allowed.shape[1]
    kept = np.zeros(len(allowed), dtype=bool)
    n_kept = 0
    for district in range(len(allowed)):
        if n_kept == n_schools:
            break
        # Those kept can each have a school of their own, so the district can
        # too exactly when it adds a pair to the matching.
        kept[district] = True
        if matching_size(allowed[kept]) > n_kept:
            n_kept += 1
        else:
            kept[district] = False
    if n_kept < n_schools:
        raise EmptyRouteSet(
            f"Only {n_kept} of the {n_schools} schools can each have a "
            f"district of its own among the {len(allowed)} districts."
        )
    return kept


def largest_first_pairs(distances: np.ndarray, places: np.ndarray) -> np.ndarray:
    """Pair the districts in turn, each with the school offering the most
    places that no district before it took, a tie going to the nearer.

    Route length plays no other part. The rule is greedy, so a district can
    find every school it may be paired with taken where some other pairing
    would give every school a district of its own.

    Args:
        distances (np.ndarray): Distance from every district to every school,
        shape (n_districts, n_schools), np.inf where a pair is barred, the
        districts in the order they take their turns.

        places (np.ndarray): Places each school offers, shape (n_schools,).

    Returns:
        np.ndarray: Whether each pair is chosen, shape (n_districts,
        n_schools), one pair in every row and every column.

    Raises:
        ValueError: If there are not as many districts as schools, or a
        district finds every school it may be paired with taken.
    """
    n_districts, n_schools = distances.shape
    if n_districts != n_schools:
        raise ValueError(
            f"Districts are paired with schools one to one, but there are "
            f"{n_districts} districts for {n_schools} schools."
        )
    pairs = np.zeros(distances.shape, dtype=bool)
    for district, row in enumerate(distances):
        free = np.isfinite(row) & ~pairs.any(axis=0)
        if not free.any():
            raise ValueError(
                f"The district taking turn {district + 1} of {n_districts} finds "
                "every school it may be paired with taken, so not every school "
                "has a district of its own."
            )
        # Most places first, the nearer winning a tie.
        order = np.lexsort((row, -np.asarray(places)))
        pairs[district, order[free[order]][0]] = True
    return pairs


def build_routes(
    districts: gpd.GeoDataFrame,
    distances: np.ndarray,
    minutes: np.ndarray,
    min_distance: float,
    max_minutes: float,
    weight: np.ndarray,
    school_seats: np.ndarray,
    shortest_only: bool = False,
    bottleneck: bool = False,
    school_places: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build the routes connecting route-eligible districts to schools.

    A route joins one district to one school, so the route set is the subset of
    (district, school) pairs lying more than `min_distance` apart and within
    `max_minutes` by bus, the district carrying some weight, and holding at
    least one seat. Distance and minutes are measured
    from the population centroid of the district, the point its students are
    sampled around, to the school, as `distances` gives it. With
    `shortest_only` each school keeps only the shortest of its routes, the
    first district winning a tie. With `bottleneck` the districts are taken
    in the frame's order, which must be
    IDACI rank order, as `first_districts` takes them for the schools holding a
    seat, and each such school keeps only its route to one of those kept, as
    `bottleneck_pairs` pairs them. Given `school_places`, the
    districts take turns in the frame's order, each keeping only its route to a
    school of its own, as `largest_first_pairs` pairs them. A school's seats
    are split between its routes in proportion to their districts' weight, by
    `apportion`, so they sum to its seats exactly; a route apportioned none is
    not built.

    Args:
        districts (gpd.GeoDataFrame): Route-eligible districts, carrying an
        "LSOA21CD" column, and "IDACI" (the rank) under `bottleneck`. The
        index of the frame is kept as `district_idx`, so it must be
        positional into the frame students were sampled from.

        distances (np.ndarray): Distance from every district's population
        centroid to every school, shape (n_districts, n_schools), in metres,
        its rows aligned with `districts`. `school_idx` is positional into its
        columns.

        minutes (np.ndarray): Minutes by bus at the weekday morning peak from
        every district's population centroid to every school, aligned with
        `distances`, NaN where not measured.

        min_distance (float): Districts are routed only to schools further away
        than this, in metres.

        max_minutes (float): Districts are routed only to schools within this
        many minutes by bus.

        weight (np.ndarray): Each district's claim on the seats of the schools
        it is routed to, shape (n_districts,). A district of no weight is not
        routed.

        school_seats (np.ndarray): Whole seats on the routes into each school,
        shape (n_schools,).

        shortest_only (bool, optional): Keep only each school's shortest
        route. Defaults to False.

        bottleneck (bool, optional): Keep only the districts `first_districts`
        keeps in the frame's order for the schools holding a seat, and each
        such school's route to the one `bottleneck_pairs` pairs it with.
        Defaults to False.

        school_places (np.ndarray | None, optional): Places each school
        offers, shape (n_schools,). Given, keep only each school's route to
        the district `largest_first_pairs` pairs it with on them. Defaults to
        None.

    Returns:
        pd.DataFrame: One row per route, with columns "route_id" (contiguous
        from 0, the route axis `fast_DAT` indexes), "district_idx", "LSOA21CD",
        "school_idx", "capacity" and "minutes".
    """
    # A pair whose minutes were not measured, NaN, is over no cap.
    routed = (
        (distances > min_distance)
        & ~(minutes > max_minutes)
        & (np.asarray(weight) > 0)[:, None]
    )
    if shortest_only:
        # A school with no route has every distance masked, and stays unrouted.
        nearest = np.where(routed, distances, np.inf).argmin(axis=0)
        routed &= np.arange(len(districts))[:, None] == nearest
    elif bottleneck:
        if not districts["IDACI"].is_monotonic_increasing:
            raise ValueError(
                "Under bottleneck the districts are taken in the frame's order, "
                "so it must be IDACI rank order."
            )
        # A school holding no seat can carry no one, so it takes no district.
        seated = np.asarray(school_seats) > 0
        if not seated.any():
            raise EmptyRouteSet("No school holds a route seat, so none is routed.")
        if not seated.all():
            print(
                f"{(~seated).sum()} of the {len(seated)} schools hold no route "
                "seat, so take no district."
            )
        kept = first_districts(routed[:, seated])
        last = np.flatnonzero(kept)[-1]
        passed = districts["LSOA21CD"].to_numpy()[:last][~kept[:last]]
        if len(passed):
            print(
                "Districts passed over for the next in order, holding no "
                "disadvantaged student or having no school beyond "
                f"{min_distance}m that those before them leave free: "
                + ", ".join(passed)
            )
        pairs = np.zeros_like(routed)
        pairs[np.ix_(kept, seated)] = bottleneck_pairs(
            np.where(routed, distances, np.inf)[np.ix_(kept, seated)]
        )
        routed &= pairs
    elif school_places is not None:
        routed &= largest_first_pairs(
            np.where(routed, distances, np.inf), school_places
        )

    # A school no district is routed to has no route to hold its seats.
    share = np.where(routed, np.asarray(weight, dtype=float)[:, None], 0.0)
    total = share.sum(axis=0)
    held = total > 0
    quotas = np.divide(
        share * school_seats, total, out=np.zeros_like(share), where=held
    )
    capacity = apportion(quotas, np.where(held, school_seats, 0))
    if (routed & (capacity == 0)).any():
        print(
            f"{(routed & (capacity == 0)).sum()} of the {routed.sum()} routes "
            f"beyond {min_distance}m would hold no seat, so are not built."
        )
    routed &= capacity > 0
    district_pos, school_idx = np.nonzero(routed)

    return pd.DataFrame(
        {
            "route_id": np.arange(len(district_pos), dtype=np.int32),
            "district_idx": districts.index.to_numpy()[district_pos].astype(np.int32),
            "LSOA21CD": districts["LSOA21CD"].to_numpy()[district_pos],
            "school_idx": school_idx.astype(np.int32),
            "capacity": capacity[district_pos, school_idx].astype(np.int32),
            "minutes": minutes[district_pos, school_idx],
        }
    )


def route_network(
    areas: gpd.GeoDataFrame,
    district_distance: np.ndarray,
    district_minutes: np.ndarray | None,
    school_scores: np.ndarray,
    school_places: np.ndarray,
    district_cohort: np.ndarray,
    decile: int = DISADVANTAGED_DECILE,
    min_distance: float = MIN_ROUTE_DISTANCE,
    max_minutes: float = MAX_ROUTE_MINUTES,
    capacity_scale: float = ROUTE_CAPACITY_SCALE,
    max_local_p8: float = MAX_LOCAL_P8,
    local_radius: float = LOCAL_RADIUS,
    round_up: bool = ROUND_UP_SEATS,
    progressivity: float = ROUTE_PROGRESSIVITY,
    linear: bool = LINEAR_PROGRESSIVITY,
    disadvantage: str = DISADVANTAGE,
    shortest_only: bool = SHORTEST_ROUTE_ONLY,
    bottleneck: bool = BOTTLENECK_ROUTES,
    largest_first: bool = LARGEST_FIRST_ROUTES,
) -> pd.DataFrame:
    """Select the route-eligible districts of `areas` and route them to schools.

    A district is route-eligible on two conditions: it sits at or below `decile`,
    and no school scoring above `max_local_p8` lies within `local_radius` of it,
    since a district a high-performing school already serves needs no transport
    past it. The local schools are read from the same population centroid the
    distance condition measures from. Under `bottleneck` every district, in
    IDACI rank order, takes the place of those at or below `decile`, and of
    the route-eligible the most deprived that can each have a school of their
    own are routed, as many as there are schools. Under `largest_first` those
    holding the most
    disadvantaged students, n_d below, as many as there are schools, a tie
    going to the lower IDACI rank.

    The disadvantaged group is untouched by the performance condition, so the
    dissimilarity index measures the same students whether their district is
    route-eligible or not. Under every `disadvantage` but "decile" it is drawn
    from IDACI scores, so the decile does not move it either.

    The routes into school s hold k * P_s * D / N seats between them, with P_s
    the Year 7 places the school offers, D the region's disadvantaged students
    and N its cohort, so at k = 1 they hold the seats the region's
    disadvantaged students would take at the school if every intake matched
    the region's mix. Those seats are split between the school's routes in
    proportion to n_d * w_d, with n_d the disadvantaged students of the route's
    district: its `disadvantaged_cohort` under "score" and "score-open", and
    its whole cohort under "decile", where every student of a district at or
    below the decile is disadvantaged. The weight w_d is (1 / D_d)^p for a
    district at IDACI decile D_d, or ((decile + 1 - D_d) / decile)^p when
    `linear`: p moves a school's seats towards the more deprived districts
    without changing how many it holds.

    Args:
        areas (gpd.GeoDataFrame): Every district, carrying "LSOA21CD", "IDACI
        Decile" and "IDACI Score" columns, and "IDACI" (the rank) under
        `bottleneck` and `largest_first`, positionally indexed in the order
        students were sampled from.

        district_distance (np.ndarray): Distance from every district's
        population centroid to every school, shape (n_districts, n_schools),
        in metres, its rows aligned with `areas`. Every distance condition
        reads it.

        district_minutes (np.ndarray | None): Minutes by bus at the weekday
        morning peak from every district's population centroid to every
        school, aligned with `district_distance`, or None where they were not
        measured, under no cap.

        school_scores (np.ndarray): Progress 8 per school, shape (n_schools,),
        aligned with the columns of `district_distance`. `load_schools` drops
        a secondary school without a published score, so such a school is
        absent from both arrays and takes no part in this condition.

        school_places (np.ndarray): Year 7 places each school offers, as
        `load_places_offered` reads them, shape (n_schools,), aligned with
        the columns of `district_distance`.

        district_cohort (np.ndarray): Expected students in the cohort of every
        district, positionally aligned with `areas`: the mean cohort
        `cohort_sizes` gives, so the route set does not change with a sample.

        decile (int, optional): Routes are built only from districts at or
        below this IDACI decile. Defaults to DISADVANTAGED_DECILE.

        min_distance (float, optional): Districts are routed only to schools
        further away than this, in metres. Defaults to MIN_ROUTE_DISTANCE.

        max_minutes (float, optional): Districts are routed only to schools
        within this many minutes by bus. Defaults to MAX_ROUTE_MINUTES.

        capacity_scale (float, optional): k, the seats on the routes into a
        school as a multiple of the disadvantaged students' fair share of its
        places. Defaults to ROUTE_CAPACITY_SCALE.

        max_local_p8 (float, optional): A district with a school scoring above
        this within `local_radius` is not route-eligible. Defaults to
        MAX_LOCAL_P8.

        local_radius (float, optional): Radius around the centroid of a district
        that its local schools are read from, in metres. Defaults to
        LOCAL_RADIUS.

        round_up (bool, optional): Round each school's route seats up rather
        than to the nearest seat, before they are split between its routes.
        Defaults to ROUND_UP_SEATS.

        progressivity (float, optional): p >= 0, how far a school's seats lean
        towards the more deprived districts: 0 is flat, p weights decile 1
        `decile`^p times decile `decile`. Defaults to ROUTE_PROGRESSIVITY.

        linear (bool, optional): Weight by the linear profile
        (decile + 1 - D) / decile rather than by 1 / D. Defaults to
        LINEAR_PROGRESSIVITY.

        disadvantage (str, optional): One of DISADVANTAGE_CHOICES, which sets
        who is disadvantaged. Defaults to DISADVANTAGE.

        shortest_only (bool, optional): Keep only each school's route to the
        nearest route-eligible district holding a disadvantaged student, which
        then holds all of the school's route seats. Defaults to
        SHORTEST_ROUTE_ONLY.

        bottleneck (bool, optional): Route only as many districts as there
        are schools, whatever their decile: the route-eligible districts
        `first_districts` keeps in IDACI rank order, so a district that cannot
        have a school of its own gives way to the next in rank. Pair each
        school with one of them by `bottleneck_pairs`, the route then holding
        all of the school's route seats. Defaults to BOTTLENECK_ROUTES.

        largest_first (bool, optional): Route only the districts holding the
        most disadvantaged students, as many as there are schools, whatever
        their decile, and pair them in turn, from the most disadvantaged
        students down, with the schools by `largest_first_pairs` on
        `school_places`, the route then holding all of the school's route
        seats. Defaults to LARGEST_FIRST_ROUTES.

    Returns:
        pd.DataFrame: The route set, as returned by `build_routes`.
    """
    if sum((shortest_only, bottleneck, largest_first)) > 1:
        raise ValueError(
            "shortest_only, bottleneck and largest_first each keep one route "
            "per school by a rule of their own, so only one may be set."
        )
    if not 1 <= decile <= 10:
        raise ValueError(f"decile must be an IDACI decile in [1, 10], got {decile}.")
    if disadvantage not in DISADVANTAGE_CHOICES:
        raise ValueError(
            f"disadvantage must be one of {DISADVANTAGE_CHOICES}, got {disadvantage!r}."
        )

    # A student is tied to their district by position, since sample_students
    # returns a positional area index, so district_idx only means anything if
    # the area frame is positionally indexed too.
    if not areas.index.equals(pd.RangeIndex(len(areas))):
        raise ValueError(
            "The area frame is not positionally indexed, so route district_idx "
            "would not line up with the area index sample_students returns."
        )

    # Misaligned scores would gate on the wrong schools, plausibly and silently.
    n_schools = district_distance.shape[1]
    if len(district_distance) != len(areas):
        raise ValueError(
            f"district_distance must align with areas: got {len(district_distance)} "
            f"rows for {len(areas)} districts."
        )
    if district_minutes is None:
        if np.isfinite(max_minutes):
            raise ValueError(
                f"A cap of {max_minutes} minutes needs the minutes by bus measured."
            )
        district_minutes = np.full(district_distance.shape, np.nan)
    if district_minutes.shape != district_distance.shape:
        raise ValueError(
            f"district_minutes must align with district_distance: got shape "
            f"{district_minutes.shape} for {district_distance.shape}."
        )
    if len(school_scores) != n_schools:
        raise ValueError(
            f"school_scores must align with district_distance: got "
            f"{len(school_scores)} scores for {n_schools} schools."
        )
    if len(school_places) != n_schools:
        raise ValueError(
            f"school_places must align with district_distance: got "
            f"{len(school_places)} places for {n_schools} schools."
        )
    if len(district_cohort) != len(areas):
        raise ValueError(
            f"district_cohort must align with areas: got {len(district_cohort)} "
            f"cohorts for {len(areas)} districts."
        )
    if max_minutes <= 0:
        raise ValueError(f"max_minutes must be positive, got {max_minutes}.")
    if capacity_scale <= 0:
        raise ValueError(f"capacity_scale must be positive, got {capacity_scale}.")
    if progressivity < 0:
        raise ValueError(f"progressivity must be non-negative, got {progressivity}.")

    cohort = np.asarray(district_cohort, dtype=float)
    disadvantaged = (
        np.where(areas["IDACI Decile"].to_numpy() <= decile, cohort, 0.0)
        if disadvantage == "decile"
        else disadvantaged_cohort(areas, cohort)
    )
    if bottleneck:
        # Every district, most deprived first: the order `first_districts`
        # takes the route-eligible in, once the conditions have excluded the
        # rest, so a district excluded gives way to the next in rank.
        deprived = areas.sort_values("IDACI", kind="stable")
        selected = f"{len(deprived)} districts in IDACI rank order"
    elif largest_first:
        # Most disadvantaged students first, a tie going to the lower IDACI
        # rank: the order the districts take their turns in.
        most = np.lexsort((areas["IDACI"].to_numpy(), -disadvantaged))
        deprived = areas.iloc[most[:n_schools]]
        selected = f"{len(deprived)} districts holding the most disadvantaged students"
    else:
        deprived = areas[areas["IDACI Decile"] <= decile]
        if deprived.empty:
            raise ValueError(
                f"No LSOA sits at or below IDACI decile {decile}, so there are "
                "no districts to build routes from."
            )
        selected = f"{len(deprived)} districts at IDACI decile {decile} or below"

    # A school above the threshold inside the radius serves the district
    # already, so no route is built from it. Carried on the district index, so
    # the eligible subset keeps the positions district_idx is read from.
    local = district_distance[deprived.index] <= local_radius
    served = pd.DataFrame(
        local & (np.asarray(school_scores) > max_local_p8), index=deprived.index
    ).any(axis=1)
    eligible = deprived[~served]
    if eligible.empty:
        raise EmptyRouteSet(
            f"Every one of the {selected} has a school scoring above Progress 8 "
            f"{max_local_p8} within {local_radius}m, so no district is "
            "route-eligible."
        )

    eligible_disadvantaged = disadvantaged[eligible.index]
    if eligible_disadvantaged.sum() == 0:
        raise EmptyRouteSet(
            f"The {len(eligible)} route-eligible districts hold no disadvantaged "
            "student, so no route can hold a seat."
        )
    district_decile = eligible["IDACI Decile"].to_numpy()
    rank = (decile + 1 - district_decile) / decile if linear else 1 / district_decile
    # Under `bottleneck` each school's one route holds all of its seats, so the
    # weight only gates on disadvantaged students, and the progressive weight,
    # on the linear profile not positive above the decile bottleneck draws
    # from, plays no part.
    weight = eligible_disadvantaged * (1 if bottleneck else rank**progressivity)
    seats = (
        capacity_scale
        * np.asarray(school_places, dtype=float)
        * disadvantaged.sum()
        / cohort.sum()
    )
    routes = build_routes(
        eligible,
        district_distance[eligible.index],
        district_minutes[eligible.index],
        min_distance,
        max_minutes,
        weight,
        (np.ceil(seats) if round_up else np.rint(seats)).astype(np.int64),
        shortest_only,
        bottleneck,
        np.asarray(school_places) if largest_first else None,
    )
    if routes.empty:
        raise EmptyRouteSet(
            f"No school lies more than {min_distance}m"
            + (
                f" and within {max_minutes} minutes by bus"
                if np.isfinite(max_minutes)
                else ""
            )
            + f" from any of the {len(eligible)} route-eligible districts on a "
            "route holding a seat "
            f"at capacity scale {capacity_scale}, so the route set is empty."
        )

    # Every school can sit within the threshold of a district, or hold no seat
    # for it beyond, leaving it out of the transport scheme entirely. That is a
    # real result at a large min_distance or a small capacity_scale rather than
    # an error, but it is silent, so it is named here. Districts the
    # performance condition excludes are counted in the summary instead, since
    # it excludes them by the dozen. Under `shortest_only` most districts are
    # nearest no school, which leaves them unrouted for that reason instead.
    # Under `bottleneck` all but as many as there are schools go unrouted,
    # and `build_routes` names those passed over for one later in rank. Under
    # `largest_first` every district is paired unless its route holds no
    # seat, which `build_routes` names.
    unrouted = eligible.loc[~eligible.index.isin(routes["district_idx"]), "LSOA21CD"]
    if len(unrouted) and not (shortest_only or bottleneck or largest_first):
        print(
            f"No secondary school beyond {min_distance}m holds a seat on a "
            "route, so no routes: " + ", ".join(unrouted)
        )

    if shortest_only:
        rule = ", each school keeping only its route to the nearest district."
    elif bottleneck:
        longest = district_distance[routes["district_idx"], routes["school_idx"]].max()
        deepest = areas["IDACI"].iloc[routes["district_idx"]].max()
        rule = (
            ", each school paired with a district of its own, the districts "
            f"reaching down to IDACI rank {deepest}, the longest route "
            f"{longest:.0f}m."
        )
    elif largest_first:
        rule = (
            ", the districts in turn from the most disadvantaged each paired "
            "with the free school offering the most places."
        )
    else:
        rule = "."
    # Under `bottleneck` every district is a candidate, so a share per
    # district says nothing of the routes.
    per_district = (
        "" if bottleneck else f", {len(routes) / len(eligible):.1f} per district"
    )
    print(
        f"{selected}, {len(eligible)} of them with no school above Progress 8 "
        f"{max_local_p8} within {local_radius}m, {len(routes)} routes to "
        f"{n_schools} secondary schools" + per_district + rule
    )
    return routes
