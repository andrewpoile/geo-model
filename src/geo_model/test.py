import numpy as np
# from numba import njit

n_students = 10
n_schools = 3
capacity = 3

students = np.arange(n_students)
schools = np.arange(n_schools)
capacity = np.full_like(schools,capacity)


def build_preference_array(
        preferrers:np.ndarray,
        preferees:np.ndarray,
) -> np.ndarray:
    return preferees[
        np.argsort(
            np.random.rand(
                len(preferrers),
                len(preferees),
            ),
            axis=1
        )
    ]


student_preferences = build_preference_array(
    students,
    schools,
)

school_priorities = build_preference_array(
    schools,
    students,
)

print(student_preferences[0])
# print(school_priorities)