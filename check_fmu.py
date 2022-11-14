from numerous.engine.system import Subsystem
from numerous.engine.system.fmu_subsystem import FMU_Subsystem

model_names = ['BouncingBall', 'Dahlquist', 'Feedthrough', 'Stair', 'VanDerPol']


class S3(Subsystem):
    def __init__(self, tag, file_name):
        super().__init__(tag)

        fmu_filename = file_name
        fmu_subsystem = FMU_Subsystem(fmu_filename, tag, debug_output=True)
        self.register_items([fmu_subsystem])


for model_name in model_names:
    system = S3(model_name, f'build_tmp/dist/{model_name}.fmu')
