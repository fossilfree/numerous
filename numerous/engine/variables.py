from dataclasses import dataclass
from enum import Enum
from typing import Any
import uuid
from functools import reduce
from operator import add


class VariableBase:
    pass


class VariableType(Enum):
    CONSTANT = 0
    PARAMETER = 1
    STATE = 2
    DERIVATIVE = 3


class OverloadAction(Enum):
    RaiseError = 0
    SUM = 1


@dataclass
class VariableDescription:
    tag: str
    type: VariableType = VariableType.PARAMETER
    initial_value: Any = None
    id: str = None
    on_assign_overload: OverloadAction = OverloadAction.RaiseError


@dataclass
class DetailedVariableDescription(VariableDescription):
    namespace: Any = None
    item: Any = None
    metadata: dict = None
    mapping: None = None
    update_counter: int = None
    allow_update: True = None


class Variable:


    def __init__(self, detailed_variable_description, base_variable=None):

        self.detailed_description = detailed_variable_description
        self.namespace = detailed_variable_description.namespace
        self.tag = detailed_variable_description.tag
        self.id = detailed_variable_description.id
        self.type = detailed_variable_description.type
        self.path = detailed_variable_description.tag
        self.alias = None
        if base_variable:

            self.value = base_variable.value
        else:
            self.value = detailed_variable_description.initial_value
        self.item = detailed_variable_description.item
        self.metadata = detailed_variable_description.metadata
        self.mapping = detailed_variable_description.mapping
        self.update_counter = detailed_variable_description.update_counter
        self.allow_update = detailed_variable_description.allow_update
        self.on_assign_overload = detailed_variable_description.on_assign_overload
        self.associated_scope = []

    def add_mapping(self, variable):
        self.mapping.append(variable)

    def extend_path(self, tag):
        self.path = tag + '.' + self.path


    def __getattribute__(self, item):
        if item == 'value':
            if self.mapping:
                return reduce(add, [x.value for x in self.mapping])
            else:
                return object.__getattribute__(self, item)
        return object.__getattribute__(self, item)

    def update_value(self, value):
        self.value = value

    @staticmethod
    def create(namespace, v_id, tag,
               v_type, value, item, metadata,
               mapping, update_counter, allow_update):
        return Variable(DetailedVariableDescription(tag=tag,
                                                    id=v_id,
                                                    type=v_type,
                                                    initial_value=value,
                                                    namespace=namespace,
                                                    item=item,
                                                    metadata=metadata,
                                                    mapping=mapping,
                                                    update_counter=update_counter,
                                                    allow_update=allow_update))

    def update(self, value):
        if self.on_assign_overload == OverloadAction.SUM:
            self.value += value
        else:
            raise ValueError('Reassignment of variable {0} not allowed'.format(self.tag))

    def __setattr__(self, key, value):
        if key == 'value' and 'update_counter' in self.__dict__:
            if self.allow_update:
                self.update_counter += 1
            else:
                if self.type == VariableType.CONSTANT:
                    raise ValueError(' It is not possible to reassign constant variable {0}'
                                     .format(self.tag))
                else:

                    raise ValueError('It is not possible to reassign variable {0}'
                                     ' in differential equation'.format(self.tag))

        object.__setattr__(self, key, value)


class _VariableFactory:

    @staticmethod
    def _create_from_variable_desc(namespace, item, var_desc):
        return Variable.create(namespace=namespace,
                               v_id="{0}_{1}_{2}_{3}".format(item.tag, namespace.tag, var_desc.tag, uuid.uuid4()),
                               tag=var_desc.tag,
                               v_type=var_desc.type,
                               value=var_desc.initial_value,
                               item=item,
                               metadata={},
                               mapping=[],
                               update_counter=0,
                               allow_update=(var_desc.type != VariableType.CONSTANT)
                               )

    @staticmethod
    def _create_from_variable_desc_unbound(initial_value, variable_description):
        v1 = Variable.create(namespace=None,
                             v_id="{0}_{1}".format(variable_description.tag, uuid.uuid4()),
                             tag=variable_description.tag,
                             v_type=variable_description.type,
                             value=initial_value,
                             item=None,
                             metadata={},
                             mapping=[],
                             update_counter=0,
                             allow_update=(variable_description.type != VariableType.CONSTANT))

        return v1
