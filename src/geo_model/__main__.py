from __future__ import annotations
 
import multiprocessing as mp
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable
import numpy as np

Student = Any
School  = Any
Route   = Any

@dataclass(frozen=True)
class Contract:
    student: Student
    school:  School
    route:   Route | None = None


students = [1,2,3,4,5,6]
schools  = [1,2,3]
routes   = [1,2]

school_capacities = [30,60,30]
route_capacities = [20,30]

contracts = [
        Contract(s,c,r)
        for s in students
        for c in schools
        for r in [*routes, None]
    ];print(contracts)

student_preferences = [
    [Contract(1,1,1),Contract(1,1,None),Contract(1,2,2),Contract(1,2,None)],
    [Contract(2,1,1),Contract(2,1,None),Contract(2,2,2),Contract(2,2,None)],
    [Contract(3,3,None),Contract(3,1,None),Contract(3,2,None)],
    [Contract(4,3,None),Contract(4,1,None),Contract(4,2,None)],
    [Contract(5,3,None),Contract(5,1,None),Contract(5,2,None)],
    [Contract(6,2,2),Contract(6,2,None),Contract(6,1,1),Contract(6,1,None)],
]

school_priorities = [
    [
        Contract(s,1,r)
        for s in students
        for r in [1,None]
    ],
    [
        Contract(s,2,r)
        for s in students
        for r in [2,None]
    ],
    [
        Contract(s,3,None)
        for s in students
    ],
]

print(school_priorities[0])