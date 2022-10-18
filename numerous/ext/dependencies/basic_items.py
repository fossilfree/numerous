from __future__ import annotations
from typing import Annotated

from numerous.ext.dependencies.basic_funcs import add_power_with_energy, Cp_W, T_az
from numerous.ext.engine_ext import Parameter, State, Constant, ScopeSpec, Module, EquationSpec, allow_implicit, ItemsSpec, create_mappings


class ControlVolume(Module):
    """
    Class implementing a control volume
    """

    tag: Annotated[str, "tag for the model"] = 'volume'

    class ControlVolumeVariables(ScopeSpec):
        P = Parameter(0)
        F = Parameter(0)
        T, T_dot = State(20)
        C = Constant(1000)

    default = ControlVolumeVariables()

    @EquationSpec(default)
    def diff(self, scope: ControlVolumeVariables):
        scope.T_dot = scope.P / scope.C


class FixedFlow(Module):
    """
        Model of a pipe using a fixed flow to propagate energy from one volume to another.
    """

    tag = 'pipe'

    class FixedFlowVariables(ScopeSpec):
        """
        Variables for Fixed Flow Model
        """
        side1_P: Annotated[Parameter, "Power assigned to side 1", "[W]"] = Parameter(0)
        side1_F: Annotated[Parameter, "Flow assigned to side 1", "[kg/s"] = Parameter(0)
        side1_T: Annotated[Parameter, "Temperature of side 1", "[degC."] = Parameter(20)
        side2_P = Parameter(0)
        side2_F = Parameter(0)
        side2_T = Parameter(20)
        F = Parameter(0)

    default = FixedFlowVariables()

    class FixedFlowItems(ItemsSpec):
        """
        Items for fixed flow model
        """
        side1: Annotated[ControlVolume, "Side 1 which this model will connect to for transferring flow and power"] = ControlVolume
        side2: Annotated[ControlVolume, "Side 2 which this model will connect to for transferring flow and power"] = ControlVolume

    items = FixedFlowItems()

    with create_mappings() as mappings:
        # Map side1_ variables to side 1 item
        items.side1.default.P += default.side1_P
        items.side1.default.F += default.side1_F
        default.side1_T = items.side1.default.T

        # Map side2_ variables to side 2 item
        items.side2.default.P += default.side2_P
        items.side2.default.F += default.side2_F
        default.side2_T = items.side2.default.T

    def __init__(self, side1: ControlVolume, side2: ControlVolume, tag=tag):
        super(FixedFlow, self).__init__(tag)

        # Assign the side1 and side2 control volumes
        self.items.side1 = side1
        self.items.side2 = side2

    @EquationSpec(default)
    def diff(self, scope:FixedFlowVariables):
        P = (scope.side1_T + T_az) * Cp_W * scope.F if scope.F <= 0 else (scope.side2_T + T_az) * Cp_W * scope.F
        scope.side1_P = -P
        scope.side2_P = P

        scope.side1_F = -scope.F
        scope.side2_F = scope.F


if __name__ == '__main__':
    cv1 = ControlVolume("a")
    cv2 = ControlVolume("b")

    ff = FixedFlow(side1=cv1, side2=cv2, tag="a_b")
