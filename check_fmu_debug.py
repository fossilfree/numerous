import numpy as np
import pandas

from numerous.engine.model import Model
from numerous.engine.simulation import Simulation
from numerous.engine.system import Subsystem
from numerous.engine.system.fmu_subsystem import FMU_Subsystem
from fmpy import read_model_description


model_names = ['VanDerPol']


class S3(Subsystem):
    def __init__(self, tag, file_name):
        super().__init__(tag)

        fmu_filename = file_name
        fmu_subsystem = FMU_Subsystem(fmu_filename, tag, debug_output=True)
        self.register_items([fmu_subsystem])


for model_name in model_names:
    description = read_model_description(f'{model_name}.fmu')
    step_size = description.defaultExperiment.stepSize
    start_time = description.defaultExperiment.startTime
    end_time = description.defaultExperiment.stopTime
    num = int((end_time - start_time) / step_size)
    system = S3('q1', f'{model_name}.fmu')
    for use_llvm in [False]:
        m1 = Model(system, use_llvm=use_llvm)
        s = Simulation(
            m1, t_start=start_time, t_stop=end_time, num=num)
        s.solve()
        ref = pandas.read_csv(f'{model_name}_ref.csv')
        print(m1.historian_df)
        print(m1.historian_df.aliases)
        for col in ref.columns:
            if col == 'time':
                continue
            else:
                print(ref[col])
                print(np.array(m1.historian_df[f'q1.q1.t1.{col}']))
                assert np.allclose(ref[col], np.array(m1.historian_df[f'q1.q1.t1.{col}']), rtol=0.2)