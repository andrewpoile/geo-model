import numpy as np
from numba import njit
from numpy.typing import NDArray


# Called with parentheses so the type checker sees the compiled function's
# signature rather than numba's decorator wrapper. The compiled kernel is
# cached on disk, so only the first process after an edit pays the compile.
@njit(cache=True)
def fast_DAT(
    student_preferences: NDArray[np.int32],
    school_priorities: NDArray[np.int32],
    school_capacities: NDArray[np.int32],
    route_capacities: NDArray[np.int32],
) -> NDArray[np.int32]:
    """Fast Deferred Acceptance with Transportation.
    Takes student preferences, school priorities, and
    school and route capacities, then outputs a matching.

    A student holds at most one seat, so the rank of the bundle each student
    holds is kept per student, and a full school finds its worst-ranked
    seated bundle by scanning its own students rather than every bundle it
    could ever be offered. Among equal ranks the lower route index gives way
    first, routeless bundles last, and the lower student index among those.

    Args:
        student_preferences (np.ndarray[student, preference, object]): Elements of the array are schools and routes.
        school_priorities (np.ndarray[school, route, student]): Elements of the array are school priority rankings.
        school_capacities (np.ndarray[school]): Elements of the array are school capacities.
        route_capacities (np.ndarray[route]): Elements of the array are route capacities.

    Returns:
        np.ndarray[student, object]: Final matching of students to schools and/or routes.
    """
    n_students, max_preferences, _ = student_preferences.shape
    n_schools = school_priorities.shape[0]
    n_routes = route_capacities.shape[0]
    matching = np.full((n_students, 2), -1, dtype=np.int32)

    student_next_preference_idx = np.zeros(n_students, dtype=np.int32)
    school_acceptance_numbers = np.zeros(n_schools, dtype=np.int32)
    route_acceptance_numbers = np.zeros(n_routes, dtype=np.int32)
    # Priority rank of the bundle each student holds, -1 while unseated.
    held_rank = np.full(n_students, -1, dtype=np.int32)

    free_students = np.arange(n_students, dtype=np.int32)
    pointer = n_students
    while pointer > 0:
        pointer -= 1
        s_id = free_students[pointer]
        s_rank = student_next_preference_idx[s_id]
        if s_rank >= max_preferences:
            continue
        student_next_preference_idx[s_id] += 1

        target_school = student_preferences[s_id, s_rank, 0]
        target_route = student_preferences[s_id, s_rank, 1]
        if target_school < 0 or target_school >= n_schools:
            free_students[pointer] = s_id
            pointer += 1
            continue

        c_rank = school_priorities[target_school, target_route, s_id]
        c_acc_num = school_acceptance_numbers[target_school]
        school_is_full = c_acc_num >= school_capacities[target_school]
        if target_route > -1:
            r_acc_num = route_acceptance_numbers[target_route]
            route_is_full = r_acc_num >= route_capacities[target_route]
        else:
            route_is_full = False
        accepted = False

        if not school_is_full and not route_is_full:
            held_rank[s_id] = c_rank
            school_acceptance_numbers[target_school] += 1
            if target_route > -1:
                route_acceptance_numbers[target_route] += 1
            matching[s_id, 0] = target_school
            matching[s_id, 1] = target_route
            accepted = True

        elif route_is_full:
            lowest_priority_student = -1
            worst_rank = -1
            for s in range(n_students):
                if (
                    matching[s, 0] == target_school
                    and matching[s, 1] == target_route
                    and held_rank[s] > worst_rank
                ):
                    worst_rank = held_rank[s]
                    lowest_priority_student = s
            eviction_required = c_rank < worst_rank
            if eviction_required:
                held_rank[lowest_priority_student] = -1
                held_rank[s_id] = c_rank
                matching[lowest_priority_student, 0] = -1
                matching[lowest_priority_student, 1] = -1
                matching[s_id, 0] = target_school
                matching[s_id, 1] = target_route
                free_students[pointer] = lowest_priority_student
                pointer += 1
                accepted = True

        else:
            lowest_priority_student = -1
            lowest_priority_route = -1
            worst_rank = -1
            worst_route = n_routes + 1
            for s in range(n_students):
                if matching[s, 0] != target_school:
                    continue
                rank = held_rank[s]
                # A routeless seat sits last on the route axis, so it is the
                # last to give way among equal ranks.
                route = matching[s, 1] if matching[s, 1] > -1 else n_routes
                if rank > worst_rank or (rank == worst_rank and route < worst_route):
                    worst_rank = rank
                    worst_route = route
                    lowest_priority_route = matching[s, 1]
                    lowest_priority_student = s
            eviction_required = c_rank < worst_rank
            if eviction_required:
                held_rank[lowest_priority_student] = -1
                held_rank[s_id] = c_rank
                # The evicted student gives up their seat on their route as
                # well as the one at the school, and the applicant takes both.
                if lowest_priority_route > -1:
                    route_acceptance_numbers[lowest_priority_route] -= 1
                if target_route > -1:
                    route_acceptance_numbers[target_route] += 1
                matching[lowest_priority_student, 0] = -1
                matching[lowest_priority_student, 1] = -1
                matching[s_id, 0] = target_school
                matching[s_id, 1] = target_route
                free_students[pointer] = lowest_priority_student
                pointer += 1
                accepted = True

        if not accepted:
            free_students[pointer] = s_id
            pointer += 1
    return matching
