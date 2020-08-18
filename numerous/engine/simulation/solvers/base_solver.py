from enum import Enum


class SolverType(Enum):
    SOLVER_IVP = 0
    NUMEROUS = 1


class BaseSolver:
    def __init__(self):
        pass
