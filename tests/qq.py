from numerous.engine.system import Item, Subsystem
from numerous.multiphysics import EquationBase, Equation
from numerous.engine.model import Model
from numerous.engine.simulation import Simulation
import numpy as np


class Item1(Item, EquationBase):
    def __init__(self, tag='item1'):
        super(Item1, self).__init__(tag)
        self.t1 = self.create_namespace('t1')
        self.add_parameter('x0', 0)
        self.add_parameter('x1', 0)
        self.t1.add_equations([self])

    @Equation()
    def eval(self, scope):
        scope.x0 = 1
        scope.x1 = 1


class Subsys1(Subsystem, EquationBase):
    def __init__(self, tag='subsys1', item1=Item1()):
        super(Subsys1, self).__init__(tag)
        self.t1 = self.create_namespace('t1')
        self.add_parameter('x2', 0)
        self.t1.add_equations([self])
        self.register_item(item1)
        self.t1.x2 += item1.t1.x0
        self.t1.x2 += item1.t1.x1


class Subsys2(Subsystem, EquationBase):
    def __init__(self, tag='subsys2', item1=Item1()):
        super(Subsys2, self).__init__(tag)
        self.t1 = self.create_namespace('t1')
        self.add_parameter('x3', 0)
        self.register_item(item1)
        self.t1.add_equations([self])
        self.t1.x3 = item1.t1.x0


class Subsys3(Subsystem, EquationBase):
    def __init__(self, tag='subsys3', subsys2=Subsys2()):
        super(Subsys3, self).__init__(tag)
        self.t1 = self.create_namespace('t1')
        self.add_parameter('x4', 0)
        self.add_parameter('x5', 0)
        self.register_item(subsys2)
        self.t1.add_equations([self])
        self.t1.x4 = subsys2.t1.x3

    @Equation()
    def eval(self, scope):
        scope.x5 = np.exp(scope.x4) * 3


class Subsys4(Subsystem, EquationBase):
    def __init__(self, tag='subsys4', subsys1=Subsys1(), subsys3=Subsys3()):
        super(Subsys4, self).__init__(tag)
        self.t1 = self.create_namespace('t1')
        self.add_parameter('x6', 0)
        self.register_items([subsys1, subsys3])
        self.t1.add_equations([self])
        self.t1.x6 += subsys1.t1.x2
        self.t1.x6 += subsys3.t1.x5


class Subsys5(Subsystem, EquationBase):
    def __init__(self, tag='subsys5', subsys4=Subsys4()):
        super(Subsys5, self).__init__(tag)
        self.t1 = self.create_namespace('t1')
        self.add_parameter('x7', 0)
        self.register_items([subsys4])
        self.t1.add_equations([self])
        self.t1.x7 = subsys4.t1.x6


class Assembly(Subsystem):
    def __init__(self, tag='assembly'):
        super().__init__(tag)
        item1 = Item1()
        subsys1 = Subsys1(item1=item1)
        subsys2 = Subsys2(item1=item1)
        subsys3 = Subsys3(subsys2=subsys2)
        subsys4 = Subsys4(subsys1=subsys1, subsys3=subsys3)
        subsys5 = Subsys5(subsys4=subsys4)
        self.register_items([item1, subsys1, subsys2, subsys3, subsys4, subsys5])


sys = Assembly()
model = Model(sys)





