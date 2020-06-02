class _DictWrapper(object):
    def __init__(self, internal_dict, value_type):
        self.value_type = value_type
        self.internal_dict = internal_dict
        self.shadow_dict = {}

    def keys(self):
        return self.shadow_dict.keys()

    def __getitem__(self, y):
        """ x.__getitem__(y) <==> x[y] """
        return self.shadow_dict[y]

    def __getattribute__(self, item):
        if item == 'shadow_dict' or item == '__setstate__' or item == '__dict__':
            return object.__getattribute__(self, item)
        if item in self.shadow_dict:
            return self.shadow_dict[item]
        return object.__getattribute__(self, item)

    def __iter__(self):
        return iter(self.shadow_dict.values())

    def values(self):
        return self.shadow_dict.values()

    def update(self, E=None, **F):
        self.internal_dict.update(E, **F)
        self.shadow_dict.update(E, **F)

    def __setitem__(self, key, value):
        if isinstance(value, self.value_type):
            self.shadow_dict[key] = value
            self.internal_dict[key] = value
        else:
            raise ValueError(
                "Object of type {0} cannot be added to internal dictionary of type {1}".format(type(value),
                                                                                               self.value_type))
