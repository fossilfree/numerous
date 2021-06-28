from numba.experimental import jitclass


@jitclass()
class Event:

    def __init__(self):
        self.name =""
        self.event_type = ""

    def chkec(self)->bool:
        pass


    def exec(self):
        pass