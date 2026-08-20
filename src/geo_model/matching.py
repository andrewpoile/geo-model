import numpy as np
from numba import njit

@njit
def fast_DAT(
        student_preferences:np.ndarray[(3,), np.int32],
        school_priorities:np.ndarray[(3,), np.int32],
        school_capacities:np.ndarray[(1,), np.int32],
        route_capacities:np.ndarray[(1,), np.int32],
) -> np.ndarray:
    """Fast Deferred Acceptance with Transportation.
    Takes student preferences, school priorities, and
    school and route capacities, then outputs a matching.

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
    assigned_students = np.full((n_schools, n_routes+1, n_students), -1, dtype=np.int32)

    free_students = np.arange(n_students, dtype=np.int32)
    pointer = n_students
    while pointer > 0:
        # print(pointer,"\n")
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
        else: route_is_full = False
        accepted = False

        if not school_is_full and not route_is_full:
            assigned_students[target_school, target_route, s_id] = c_rank
            school_acceptance_numbers[target_school] += 1
            if target_route > -1:
                route_acceptance_numbers[target_route] += 1
            matching[s_id, 0] = target_school
            matching[s_id, 1] = target_route
            accepted = True

        elif route_is_full:
            lowest_priority_student = assigned_students[target_school, target_route].argmax()
            eviction_required = (
                c_rank < school_priorities[target_school, target_route, lowest_priority_student]
            )
            accepted = False
            if eviction_required:
                assigned_students[target_school, target_route, lowest_priority_student] = -1
                assigned_students[target_school, target_route, s_id] = c_rank
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
            for r in range(n_routes + 1):
                for s in range(n_students):
                    rank = assigned_students[target_school, r, s]
                    if rank > worst_rank:
                        worst_rank = rank
                        lowest_priority_route = r
                        lowest_priority_student = s
            # lowest_priority_column = assigned_students[target_school].max(axis=1)
            # lowest_priority_route = lowest_priority_column.argmax()
            # lowest_priority_student = assigned_students[target_school, lowest_priority_route].argmax()
            eviction_required = (
                c_rank < worst_rank
            )
            accepted = False
            if eviction_required:
                assigned_students[target_school, lowest_priority_route, lowest_priority_student] = -1
                assigned_students[target_school, target_route, s_id] = c_rank
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