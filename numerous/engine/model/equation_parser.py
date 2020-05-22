
import ast
import re
import numpy as np

# 3. Compute compiled_eq and compiled_eq_idxs, the latter mapping
# self.synchronized_scope to compiled_eq (by index)
class Equation_Parser():

    def __init__(self):
        pass

    def parse(self,model):
        compiled_equations_idx = []
        compiled_eq = []
        compiled_equations_ids = []  # As equations can be non-unique, but they should?
        for tt in model.synchronized_scope.keys():
            eq_text = ""
            eq_id = ""
            # id of eq in list of eq)
            if model.equation_dict[tt]:
                if model.equation_dict[tt][0].id in compiled_equations_ids:
                    compiled_equations_idx.append(compiled_equations_ids.index(model.equation_dict[tt][0].id))
                    continue
                lines = model.equation_dict[tt][0].lines.split("\n")
                eq_id = model.equation_dict[tt][0].id
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

                # eq_text = eq_text.replace("@Equation()", "@guvectorize(['void(float64[:])'],'(n)',nopython=True)")
                eq_text = eq_text.replace("@Equation()", "@simple_vectorize")
                # eq_text = eq_text.replace("self,", "global_variables,")
                eq_text = eq_text.replace("self,", "")
                eq_text = eq_text.strip()
                idx = eq_text.find('\n') + 1
                # spaces_len = len(eq_text[idx:]) - len(eq_text[idx:].lstrip())
                # eq_text = eq_text[:idx] + " " * spaces_len + 'import numpy as np\n' + " " * spaces_len \
                #           + 'from numba import njit\n' + eq_text[idx:]
                str_list = []
                for line in eq_text.splitlines():
                    str_list.append('   ' + line)
                eq_text_2 = '\n'.join(str_list)

                eq_text = eq_text_2 + "\n   return eval"
                # eq_text = "def test():\n   from numba import guvectorize\n   import numpy as np\n"+eq_text
                eq_text = "def eq_body():\n   from numerous.engine.model.simple_vectorizer import simple_vectorize\n   import numpy as np\n" + eq_text
            else:
                eq_id = "empty_equation"
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

        return np.array(compiled_eq),np.array(compiled_equations_idx,dtype=np.int64)

