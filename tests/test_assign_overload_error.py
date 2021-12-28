from numerous.engine.system import Item, Subsystem, ItemPath
from numerous.multiphysics import EquationBase, Equation
import pytest


class Item1(Item, EquationBase):
    def __init__(self, tag='item1'):
        super(Item1, self).__init__(tag)
        self.t1 = self.create_namespace('t1')
        self.add_parameter('T1', 0)
        self.t1.add_equations([self])

    @Equation()
    def eval(self, scope):
        scope.T1 = 1


class Item2(Item, EquationBase):
    def __init__(self, tag='item2'):
        super(Item2, self).__init__(tag)
        self.t1 = self.create_namespace('t1')
        self.add_parameter('T2', -1)
        self.t1.add_equations([self])

    @Equation()
    def eval(self, scope):
        scope.T2 = 1



@pytest.fixture
def system1():
    class System(Subsystem, EquationBase):
        def __init__(self, tag='system1_test_assign', item1=object, item2=object):
            super(System, self).__init__(tag)
            self.register_items([item1, item2])

    return System(item1=Item1(), item2=Item2())


# This tests if an error is raised upon overwriting Item1.t1.T1 with Item2.t1.T2
def test_assign_overload_error(system1):
    with pytest.raises(ValueError, match=r".*Variable already have mapping.*"):
        system1.get_item(ItemPath('system1_test_assign.item1')).t1.T1 = system1.get_item(ItemPath('system1_test_assign.item2')).t1.T2
        system1.get_item(ItemPath('system1_test_assign.item1')).t1.T1 += system1.get_item(ItemPath('system1_test_assign.item2')).t1.T2
