class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Watcher(metaclass=Singleton):
    declarations = []

    def add_watched_object(self, object_to_watch):
        self.declarations.append(object_to_watch)

    def dewatch_object(self, watched_object):

        ix = self.declarations.index(watched_object)
        if ix <0:
            raise ValueError('Object cannot be dewatched - it hasnt been watched in the first place.')
        self.declarations.pop(ix)

    def finalize(self):
        print(self.declarations)
        if len(self.declarations) > 0:
            raise ValueError("Not all watched objects finalized!")

watcher = Watcher()

class WatchedObject_:

    def __init__(self, *args, **kwargs):
        self.watcher = watcher
        self.watcher.add_watched_object(self)
        self.dangling = True

    def finalize(self):
        pass

    def attach(self):
        self.dangling = False
        self.watcher.dewatch_object(self)



class WatchObjectMeta(type):

    def __init__(self, class_name: str, base_classes: tuple, __dict__: dict, **kwargs):
        super(WatchObjectMeta, self).__init__(class_name, base_classes, __dict__, **kwargs)

        for class_var, value in __dict__.items():

            if isinstance(value, WatchedObject_):
                value.attach()




class WatchedObject(WatchedObject_, metaclass=WatchObjectMeta):
    ...