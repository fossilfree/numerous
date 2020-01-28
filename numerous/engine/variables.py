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


@dataclass
class DetailedVariableDescription(VariableDescription):
    namespace: Any = None
    item: Any = None
    metadata: dict = None
    mapping: None = None
    update_counter: int = None
    allow_update: True = None


class MappedValue(object):
    def __init__(self):
        self.mapping = []
        self.sum_mapping = []
        self.special_mapping = False

    def add_mapping(self, variable):

        if not self.special_mapping:
            if variable.id == self.id:
                raise RecursionError("Variable {0} cannot be mapped to itself",self.id)
            self.mapping.append(variable)
        self.special_mapping = False

    def add_sum_mapping(self, variable):
        self.sum_mapping.append(variable)

    def __iadd__(self, other):
        if isinstance(other, Variable):
            if self.mapping:
                raise ValueError('It is not possible to add a summation to {0}. Variable already have mapping'
                                 ''.format(self.tag))
            else:
                self.add_sum_mapping(other)
                self.special_mapping = True
                return self
        else:
            object.__iadd__(self, other)

    def get_value(self):
        if self.mapping:
            return reduce(add, [x.value for x in self.mapping])
        if self.sum_mapping:
            return self.value + reduce(add, [x.value for x in self.sum_mapping])
        else:
            return self.value

class VariablePath:

    def __init__(self, tag, id):

        self.path = {id: tag}
        self.used_id_pairs = []

    def __iter__(self):
        return iter(self.path.values())

    def extend_path(self, current_id, new_id, new_tag):
        if not (current_id+new_id in self.used_id_pairs):
            if new_id in  self.path:
                self.path[new_id].extend([new_tag+'.'+x for x in self.path[current_id]])
            else:
                self.path.update({new_id: [new_tag+'.'+x for x in self.path[current_id]]})
            self.used_id_pairs.append(current_id+new_id)

class Variable(MappedValue):

    def __init__(self, detailed_variable_description, base_variable=None):

        super().__init__()
        self.detailed_description = detailed_variable_description
        self.namespace = detailed_variable_description.namespace
        self.tag = detailed_variable_description.tag
        self.id = detailed_variable_description.id
        self.type = detailed_variable_description.type
        self.path = VariablePath([detailed_variable_description.tag],self.id)
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
        self.associated_scope = []


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
