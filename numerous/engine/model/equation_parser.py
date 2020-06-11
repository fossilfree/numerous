import ast
import re
import numpy as np
import typing as tp


# 3. Compute compiled_eq and compiled_eq_idxs, the latter mapping
# self.synchronized_scope to compiled_eq (by index)
class Equation_Parser():

    def __init__(self):
        pass

    def get_complex_id(self, eq_dict):
        id = ""
        for i in range(len(eq_dict)):
            id += eq_dict[i].id
        return id

    def parse(self, model):
        compiled_equations_idx = []
        compiled_eq = []
        compiled_eq_m = []
        compiled_equations_ids = []  # As equations can be non-unique, but they should?
        for tt in model.synchronized_scope.keys():
            eq_text = ""
            eq_id = ""
            # id of eq in list of eq)
            if model.equation_dict[tt][0]:
                eq_text_2 = ""
                eq = model.equation_dict[tt][0]
                eq_out_m = model.equation_dict[tt][1]
                if self.get_complex_id(eq) in compiled_equations_ids:
                    compiled_equations_idx.append(compiled_equations_ids.index(self.get_complex_id(eq)))
                    continue
                eq_id = self.get_complex_id(eq)
                for i in range(len(eq)):
                    lines = eq[i].lines.split("\n")

                    non_empty_lines = [line for line in lines if line.strip() != ""]

                    string_without_empty_lines = ""
                    for line in non_empty_lines:
                        string_without_empty_lines += line + "\n"

                    eq_text = string_without_empty_lines + " \n"

                    eq_text = eq_text.replace("global_variables)", ")")
                    for i, tag in enumerate(model.global_variables_tags):
                        ##TODO write coments to regex
                        p = re.compile(r"(?<=global_variables)\." + tag + r"(?=[^\w])")
                        eq_text = p.sub("[" + str(i) + "]", eq_text)
                    for i, var in enumerate(model.synchronized_scope[tt].variables.values()):
                        ##TODO write coments to regex
                        p = re.compile(r"(?<=scope)\." + var.tag + r"(?=[^\w])")
                        eq_text = p.sub("[" + str(i) + "]", eq_text)

                    ##TODO write coments to regex

                    p = re.compile(r" +def +\w+(?=\()")
                    eq_text = p.sub("def eval", eq_text)

                    eq_text = eq_text.replace("@Equation()", "@simple_vectorize")
                    eq_text = eq_text.replace("self,", "")
                    eq_text = eq_text.strip()
                    idx = eq_text.find('\n') + 1

                    str_list = []
                    for line in eq_text.splitlines():
                        str_list.append('   ' + line)
                    if eq_text_2:
                        eq_text_2 += '\n'
                        eq_text_2 += '\n'.join(str_list[2:])
                    else:
                        eq_text_2 = '\n'.join(str_list)

                    eq_text = eq_text_2 + "\n   return eval"
                    # eq_text = "def test():\n   from numba import guvectorize\n   import numpy as np\n"+eq_text
                    eq_text = "def eq_body():\n   from numerous.engine.model.simple_vectorizer import simple_vectorize\n   import numpy as np\n" + eq_text
            else:
                eq_id = "empty_equation"
                eq_out_m = 0
                if eq_id in compiled_equations_ids:
                    compiled_equations_idx.append(compiled_equations_ids.index(eq_id))
                    continue
                eq_text = "def eq_body():\n   def eval(self,scope):\n      pass\n   return eval"

            tree = ast.parse(eq_text, mode='exec')
            code = compile(tree, filename='test', mode='exec')
            namespace = {}
            exec(code, namespace)
            compiled_equations_idx.append(len(compiled_equations_ids))
            compiled_equations_ids.append(eq_id)
            compiled_eq.append(list(namespace.values())[1]())
            compiled_eq_m.append(eq_out_m)

        return np.array(compiled_eq), np.array(compiled_equations_idx, dtype=np.int64), np.argsort(
            np.array(compiled_eq_m))

    @staticmethod
    def parse_non_numba_function(function: tp.Callable, decorator_type: str, ) -> tp.Callable:
        lines = function.lines.split("\n")
        non_empty_lines = [line for line in lines if line.strip() != ""]

        string_without_empty_lines = ""
        for line in non_empty_lines:
            string_without_empty_lines += line + "\n"

        function_text = string_without_empty_lines + " \n"

        p = re.compile(r" +def +\w+(?=\()")
        function_text = p.sub("def eval", function_text)

        p = re.compile(decorator_type)
        function_text = p.sub("", function_text)
        function_text = function_text.strip()

        str_list = []
        for line in function_text.splitlines():
            str_list.append('   ' + line)
        eq_text_2 = '\n'.join(str_list)

        function_text = eq_text_2 + "\n   return eval"
        function_text = "def eq_body():\n   from numba import njit\n   import numpy as np\n" + function_text

        tree = ast.parse(function_text, mode='exec')
        code = compile(tree, filename='test', mode='exec')
        namespace = {}
        exec(code, namespace)

        return list(namespace.values())[1]()

    @staticmethod
    def create_numba_iterations(numba_model_class, list_of_functions: tp.List[tp.Callable], callable_method_name: str,
                                internal_method_name: str, iteration_block: tp.Callable, callable_parameters: str,
                                map_sorting=None) -> None:
        code_as_string = ""
        if len(list_of_functions) > 0:
            if map_sorting is not None:
                if len(map_sorting) > 0:
                    for j in map_sorting:
                        method_name = internal_method_name + str(j)
                        setattr(numba_model_class, method_name, list_of_functions[j])
                        code_as_string += iteration_block(method_name, j)
                    code_as_string = "def eval():\n   def " + callable_method_name + "(self," + callable_parameters + "):\n" \
                                     + code_as_string \
                                     + "   return " + callable_method_name
            else:
                for i, function in enumerate(list_of_functions):
                    method_name = internal_method_name + str(i)
                    setattr(numba_model_class, method_name, function)
                    code_as_string += iteration_block(method_name, i)
                code_as_string = "def eval():\n   def " + callable_method_name + "(self," + callable_parameters + "):\n" \
                                 + code_as_string \
                                 + "   return " + callable_method_name
        else:
            code_as_string = "def eval():\n   def " + callable_method_name \
                             + "(self," + callable_parameters + "):\n      pass\n   return " + callable_method_name

        tree = ast.parse(code_as_string, mode='exec')
        code = compile(tree, filename='test', mode='exec')
        namespace = {}
        exec(code, namespace)
        setattr(numba_model_class, callable_method_name, list(namespace.values())[1]())
