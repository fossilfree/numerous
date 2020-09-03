from enum import Enum


class SolverType(Enum):
    SOLVER_IVP = 0
    NUMEROUS = 1


solver_types = [SolverType.NUMEROUS, SolverType.SOLVER_IVP]


class BaseSolver:
    def __init__(self):
        pass
