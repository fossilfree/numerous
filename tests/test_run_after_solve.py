import pytest

from numerous.engine.model import Model
from numerous.engine.simulation import Simulation
from numerous.engine.system import Subsystem


class SystemTest(Subsystem):

    def __init__(self, tag):
        super().__init__(tag)
        self.run_after_solve = ['run_after', 'run_after_2']

    def run_after(self):
        print(self.tag)

    def run_after_2(self):
        print(f'{self.tag}2')


class SystemOuterTest(Subsystem):
    def __init__(self, tag):
        super().__init__(tag)
        self.run_after_solve = ['run_after', 'run_after_2']
        o_s = [SystemTest('test')]
        self.register_items(o_s)

    def run_after(self):
        print(self.tag)

    def run_after_2(self):
        print(f'{self.tag}2')


@pytest.mark.parametrize("use_llvm", [True, False])
def test_run_after(use_llvm, capsys):
    system = SystemTest('test')
    s = Simulation(Model(system, use_llvm=use_llvm))

    s.solve()
    captured = capsys.readouterr()
    assert captured.out == 'test\ntest2\n'


@pytest.mark.parametrize("use_llvm", [True, False])
def test_run_after_complex(use_llvm, capsys):
    system = SystemOuterTest('outer')
    s = Simulation(Model(system, use_llvm=use_llvm))

    s.solve()
    captured = capsys.readouterr()
    assert captured.out == 'outer\nouter2\ntest\ntest2\n'
