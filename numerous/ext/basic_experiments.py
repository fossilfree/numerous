class Test:

    def __new__(cls, *args, **kwargs):
        print('new')
        obj = object.__new__(cls)
        print('obj created')
        obj.attr_ = 1
        return obj

    def __init__(self):
        print(self.attr_)
        print('init')

Test()