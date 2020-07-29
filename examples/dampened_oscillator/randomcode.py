import numpy as np


class Randomcode():
    def __init__(self, max_dependency=20, max_lines=1):
        self.operators = ['+', '-', '*', '/']
        self.max_dependency = max_dependency
        self.max_lines = max_lines
        self.functions = [self.no, self.sin, self.cos]
        # , self.plus, self.minus, self.multiply, self.divide

    def no(self, x):
        return f'{x}'

    def exp(self, x):
        return f'np.exp({x})'

    def log(self, x):
        return f'np.log({x})'

    def sin(self, x):
        return f'np.sin({x})'

    def cos(self, x):
        return f'np.cos({x})'

    def power(self, x, k=1.1):
        return f'{x}**{k}'

    def wrap(self, lines):
        header = 'import numpy as np\ndef fun(x):\n'
        indent = '    '

        with open("test_fun.py", 'w') as f:
            f.writelines(header)
            for line in lines:
                f.writelines(f'{indent}{line}\n')

            f.writelines(f'{indent}return x\n')
            # f.writelines(f'function(np.random.uniform(0,1,{self.max_lines+1}))')

    def generate(self):
        i = 0
        lines = []
        special = []

        while True:
            rand_variables = []
            p_flag = False
            newline = f"x[{i}]="
            const = np.random.rand(1)[0] > 0.6
            if i == 0 or const:
                newline += f'{np.random.uniform(-1, 1, 1)[0]}'
            elif i > 0 and not const:

                rand_variables = np.random.randint(0, i, size=np.random.randint(1, self.max_dependency, 1)[0])

                for q in rand_variables[:-1]:
                    k = np.random.randint(0, len(self.functions), 1)[0]
                    p = np.random.rand(1)[0]
                    o = np.random.randint(0, len(self.operators), 1)[0]
                    m = np.random.rand(1)[0]
                    c = np.random.rand(1)[0]

                    if p > 0.8:
                        if not p_flag:
                            newline += '('
                            p_flag = True

                    if m > 0.7:
                        var = f'{c}*x[{q}]'
                    else:
                        var = f'x[{q}]'
                    fun = self.functions[k](var)
                    operator = self.operators[o]
                    suffix = ''
                    prefix = ''
                    if operator == '/':
                        if p_flag:
                            suffix = '('
                            prefix = ')'
                            special.append(i)

                    newline += fun + prefix + operator + suffix
                k = np.random.randint(0, len(self.functions), 1)[0]
                lastvar = f'x[{rand_variables[-1]}]'
                newline += self.functions[k](lastvar)
                if p_flag:
                    newline += ')'
            lines.append(newline)
            i += 1
            if i > self.max_lines:
                break
        self.wrap(lines)
        return lines


if __name__ == "__main__":
    r = Randomcode()
    r.wrap(r.generate())


