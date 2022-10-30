
class AnotherManagerContextActiveException(Exception):
    pass

class NoManagerContextActiveException(Exception):
    pass

class ActiveContextManager:
    _instance = None

    def __new__(class_, *args, **kwargs):
        if not isinstance(class_._instance, class_):
            class_._instance = object.__new__(class_, *args, **kwargs)
        return class_._instance

    def __init__(self):
        self._active_manager_context = None

    def get_active_manager_context(self, ignore_no_context=False):
        if not ignore_no_context and not self.is_active_manager_context_set():
            raise NoManagerContextActiveException('No active context manager!')

        return self._active_manager_context

    def is_active_manager_context_set(self):
        return self._active_manager_context is not None

    def set_active_manager_context(self, context_manager):
        if self.is_active_manager_context_set():
            raise AnotherManagerContextActiveException('Another context active!')

        self._active_manager_context = context_manager

    def is_active_manager_context(self, context_manager):
        return self.get_active_manager_context(ignore_no_context=True) == context_manager

    def clear_active_manager_context(self, context_manager):
        if self.is_active_manager_context(context_manager):
            self._active_manager_context = None
        else:
            raise ValueError(f'Trying to clear different context manager, trying to clear: {context_manager}, but active one is: {self._active_manager_context}')


class SubsystemContextManager(ActiveContextManager):
    pass

_active_subsystem = SubsystemContextManager()

class MappingsContextManager(ActiveContextManager):
    pass

_active_mappings = MappingsContextManager()

class DeclarativeContextManager(ActiveContextManager):
    pass

_active_declarative = DeclarativeContextManager()