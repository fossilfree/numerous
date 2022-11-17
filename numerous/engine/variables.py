from dataclasses import dataclass
from enum import Enum
from typing import Any
import uuid
from functools import reduce
from operator import add
from numerous.utils.logger_levels import LoggerLevel


class VariableBase:
    pass


class VariableType(Enum):
    CONSTANT = 0
    PARAMETER = 1
    STATE = 2
    DERIVATIVE = 3
    PARAMETER_SET = 4
    TMP_PARAMETER = 5
    TMP_PARAMETER_SET = 6
    CALCULATED_VARIABLE = 7


class OverloadAction(Enum):
    RaiseError = 0
    SUM = 1


@dataclass
class VariableDescription:
    tag: str
    type: VariableType = VariableType.PARAMETER
    initial_value: [float, int] = None
    id: str = None
    variable_idx: int = 0
    logger_level: LoggerLevel = LoggerLevel.ALL
    alias: str = None
    update: bool = False
    global_var: bool = False
    global_var_idx: int = -1


@dataclass
class DetailedVariableDescription(VariableDescription):
    namespace: Any = None
    item: Any = None
    metadata: dict = None
    mapping: None = None
    update_counter: int = None
    allow_update: True = None


class MappedValue(object):
    def __init__(self, id):
        self.id = str(id).replace("-", "_")
        self.mapping = None
        self.sum_mapping = []
        self.special_mapping = False
        self.addself = False
        self.logger_level = None
        self.model = None
        self.llvm_idx = None

    def add_mapping(self, variable):
        if not self.special_mapping:
            if variable.id == self.id:
                raise RecursionError("Variable {0} cannot be mapped to itself", self.id)
            self.mapping = variable
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

    def __get_value(self, ids):
        if self.id in ids:
            return self.value
        else:
            if self.mapping:
                return self.mapping.get_value()
            if self.sum_mapping:
                ids.append(self.id)
                return reduce(add, [x.__get_value(ids) for x in self.sum_mapping])

            else:
                return self.value

    def get_value(self):
        if self.mapping:
            return self.mapping.get_value()
        if self.sum_mapping:
            return reduce(add, [x.__get_value([self.id]) for x in self.sum_mapping])

        else:
            return self.value

    def write_variable(self, value, llvm_idx):
        raise NotImplementedError


class VariablePath:

    def __init__(self, tag, id):
        self.path = {id: tag}
        self.primary_path = tag
        self.used_id_pairs = []

    def __iter__(self):
        return iter(self.path.values())

    def extend_path(self, current_id, new_id, new_tag):
        if not (current_id + new_id in self.used_id_pairs):
            if new_id in self.path:
                self.path[new_id].extend([new_tag + '.' + x for x in self.path[current_id]])
                self.primary_path = new_tag + '.' + self.path[current_id][-1]
            else:
                self.path.update({new_id: [new_tag + '.' + x for x in self.path[current_id]]})
                self.primary_path = new_tag + '.' + self.path[current_id][-1]
            self.used_id_pairs.append(current_id + new_id)


class SetOfVariables:
    def __init__(self, tag, item_tag, ns_tag):
        self.tag = tag
        self.id = "SET" + str(uuid.uuid4()).replace("-", "_")
        self.variables = {}
        self.mapping = []
        self.sum_mapping = []
        self.item_tag = item_tag
        self.ns_tag = ns_tag
        self.size = 0
        self.global_var = False
        self.global_var_idx = False
        self.eq_used = []

    def get_size(self):
        return self.size

    def add_eq_used(self, eq_name: str):
        self.eq_used.append(eq_name)

    def get_path_dot(self):
        result = list(self.variables.values())[0].get_path_dot()

        return result[:result.find(self.item_tag)] + self.item_tag + "." + self.ns_tag + "." + self.tag

    def add_variable(self, variable):
        self.variables.update({variable.id: variable})
        variable.set_var = self
        if variable.sum_mapping:
            self.sum_mapping.append(variable.sum_mapping)
        if variable.mapping:
            self.mapping.append(variable.mapping)
        self.size += 1

    def get_var_by_idx(self, i):
        return next(var for var in self.variables.values() if var.set_var_ix == i)

    def __iter__(self):
        return iter(self.variables.values())


class Variable(MappedValue):

    def __init__(self, detailed_variable_description, base_variable=None):

        super().__init__(detailed_variable_description.id)
        self.detailed_description = detailed_variable_description
        self.namespace = detailed_variable_description.namespace
        self.tag = detailed_variable_description.tag
        self.type = detailed_variable_description.type
        self.path = VariablePath([detailed_variable_description.tag], self.id)
        self.paths = []
        self.global_var = detailed_variable_description.global_var
        self.global_var_idx = detailed_variable_description.global_var_idx
        self.alias = None
        self.set_var = None
        self.set_var_ix = None
        self.set_namespace = None
        self.size = 0

        self.temporary_variable = False
        if base_variable:
            self.value = base_variable.value
        else:
            self.value = detailed_variable_description.initial_value
        self.item = detailed_variable_description.item
        self.metadata = detailed_variable_description.metadata
        self.mapping = detailed_variable_description.mapping
        self.update_counter = detailed_variable_description.update_counter
        self.allow_update = detailed_variable_description.allow_update
        self.logger_level = detailed_variable_description.logger_level
        self.associated_scope = []
        self.idx_in_scope = []
        self.eq_used = []
        self.top_item = None
        self.used_in_equation_graph = False

    def get_path_dot(self):
        if self.top_item:
            return self.path.path[self.top_item][0]
        else:
            return list(self.path.path.values())[0][0]

    def add_eq_used(self, eq_name: str):
        self.eq_used.append(eq_name)

    def update_value(self, value):
        self._value = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        try:
            float(value)
        except (ValueError, TypeError):
            raise ValueError(f"Only numeric values allowed in variables (attempted to set value='{value}' in "
                             f"{self.id})")
        self._value = value
        if self.llvm_idx is not None:
            self.write_variable(value, self.llvm_idx)

    def update_set_var(self, set_var, set_namespace):
        if not self.set_var:
            self.set_var = set_var
            self.set_namespace = set_namespace
            print('path: ', self.get_path_dot())
            print('set_var: ', set_var)
            print('ns: ', set_namespace.tag)
        else:
            if self.set_var != set_var:
                print(self.set_var, ' ', set_var)
                raise ValueError(f'Setvar for {self.id} already set!')

    @staticmethod
    def create(namespace, v_id, tag,
               v_type, value, item, metadata,
               mapping, update_counter, allow_update, logger_level, variable_idx, alias, global_var, global_var_idx):
        return Variable(DetailedVariableDescription(tag=tag,
                                                    id=v_id,
                                                    type=v_type,
                                                    initial_value=value,
                                                    namespace=namespace,
                                                    item=item,
                                                    metadata=metadata,
                                                    mapping=mapping,
                                                    update_counter=update_counter,
                                                    allow_update=allow_update,
                                                    logger_level=logger_level,
                                                    variable_idx=variable_idx,
                                                    alias=alias,
                                                    global_var=global_var,
                                                    global_var_idx=global_var_idx))

    def empty_variable(self):
        self.detailed_description = None
        self.namespace = None
        self.tag = None
        self.type = None
        self.path = None
        self.paths = None
        self.global_var = None
        self.global_var_idx = None
        self.alias = None
        self.set_var = None
        self.set_var_ix = None
        self.set_namespace = None
        self.size = 0
        self.llvm_idx = None
        self.temporary_variable = False
        self.item = None
        self.metadata = None
        self.mapping = None
        self.update_counter = None
        self.allow_update = None
        self.logger_level = None
        self.associated_scope = None
        self.idx_in_scope = None
        self.top_item = None
        self.used_in_equation_graph = False
        self.value = 0


class _VariableFactory:

    ##TODO remove recreation of var description here. It is duplicated inside the item
    @staticmethod
    def _create_from_variable_desc(namespace, item, var_desc):
        return Variable.create(namespace=namespace,
                               v_id="{0}_{1}_{2}_{3}".format(item.tag, namespace.tag, var_desc.tag, uuid.uuid4()),
                               tag=var_desc.tag,
                               v_type=var_desc.type,
                               value=var_desc.initial_value,
                               item=item,
                               metadata={},
                               mapping=None,
                               update_counter=0,
                               allow_update=(var_desc.type != VariableType.CONSTANT),
                               logger_level=var_desc.logger_level,
                               variable_idx=var_desc.variable_idx,
                               alias=var_desc.alias,
                               global_var=False,
                               global_var_idx=-1
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
                             mapping=None,
                             update_counter=0,
                             allow_update=(variable_description.type != VariableType.CONSTANT),
                             logger_level=variable_description.logger_level,
                             alias=variable_description.alias,
                             variable_idx=0,
                             global_var=variable_description.global_var,
                             global_var_idx=variable_description.global_var_idx
                             )

        return v1
