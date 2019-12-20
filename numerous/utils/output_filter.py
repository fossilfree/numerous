class OutputFilter:

    def __init__(self, only_aliases=False, level=None,list_to_save = None):
        self.only_aliases = only_aliases
        self.level = level
        self.list_to_save = list_to_save if list_to_save else []

    def filter_function(self, variable):
        result = True
        if variable[0] in self.list_to_save:
            return True
        if self.only_aliases and not variable[1].alias:
            result = False

        if self.level and variable[1].item.level > self.level:
            result = False
        return result

    def filter_varaibles(self, variables):
        return dict(filter(self.filter_function, variables.items()))
