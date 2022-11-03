from .context_managers import _active_declarative
class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Watcher:
    declarations = []

    def add_watched_object(self, object_to_watch):
        #print('watch: ', object_to_watch)
        self.declarations.append(object_to_watch)

    def dewatch_object(self, watched_object):

        try:
            ix = self.declarations.index(watched_object)
        except ValueError:
            still_watched = "\n".join([str(d) for d in self.declarations])
            raise ValueError(f'Object {watched_object} cannot be dewatched - it hasnt been watched in the first place. List of actually watched objects:\n'+still_watched)
        #print('\ndewatching: ', watched_object)
        self.declarations.pop(ix)

    def finalize(self):
        #print(self.declarations)
        if len(self.declarations) > 0:
            still_watched = "\n".join([str(d) for d in self.declarations])
            raise ValueError("Not all watched objects finalized!\n\n"+still_watched)


class WatchedObject_:

    def __init__(self, *args, **kwargs):

        #self._watcher = _active_declarative.get_active_manager_context()
        #self._watcher.add_watched_object(self)
        #print('added myself: ', self)
        self._dangling = True

    def attach(self):
        self._dangling = False
        #self._watcher.dewatch_object(self)
        #print('dewatched myself: ', self)


    def finalize(self):
        self.attach()


def get_watcher():
    if _active_declarative.is_active_manager_context_set():
        watcher = _active_declarative.get_active_manager_context()
        created = False
    else:
        watcher = Watcher()
        _active_declarative.set_active_manager_context(watcher)
        created = True
    return watcher, created

def clear_watcher(watcher):
    _active_declarative.clear_active_manager_context(watcher)

class WatchObjectMeta(type):
    ...
    """
    def __new__(cls, *args, **kwargs):
        print('start: ',args)

        watcher, created = get_watcher()
        print("created: ", created)
        new_cls = super(WatchObjectMeta, cls).__new__(cls, *args, **kwargs)

        if created:
            print('finalizing')
            for name, item in new_cls.__dict__.items():

                if isinstance(item, WatchedObject_):
                    print(item)

                    #watcher.dewatch_object(item)
                    item.attach()

            #watcher.finalize()

            clear_watcher(watcher)

        print('end')

        return new_cls


    def __init__(self, class_name: str, base_classes: tuple, __dict__: dict, **kwargs):

        print('init')
        super(WatchObjectMeta, self).__init__(class_name, base_classes, __dict__, **kwargs)
        print('init end')

    def __call__(cls, *args, **kwargs):
        watcher, created = get_watcher()
        print('call')
        instance = super(WatchObjectMeta, cls).__call__(*args, **kwargs)

        if created:
            clear_watcher(watcher)
        print('call end')
        return instance"""


class WatchedObject(WatchedObject_, metaclass=WatchObjectMeta):
    ...