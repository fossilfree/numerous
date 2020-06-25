import inspect
from enum import IntEnum, unique
import ast#, astor
from textwrap import dedent
from numerous.engine.model.graph import Graph

op_sym_map = {ast.Add: '+', ast.Sub: '-', ast.Div: '/', ast.Mult: '*', ast.Pow: '**', ast.USub: '*-1'}

def get_op_sym(op):
    return op_sym_map[type(op)]

@unique
class NodeTypes(IntEnum):
    OP=0
    VAR=1
    ASSIGN=2

def attr_ast(attr):
    attr_ = attr.split('.')
    # print(attr_)
    if len(attr_) >1:
        prev = None
        attr_str = attr_[-1]
        attr_=attr_[:-1]
        for a in attr_:
            if not prev:
                prev = ast.Name(id=a)
            else:
                prev = ast.Attribute(attr=a, value=prev)

        attr_ast = ast.Attribute(attr=attr_str, value=prev)
    else:
        # print('attr_[0]')
        # print(attr_[0])
        attr_ast = ast.Name(id=attr_[0])
    return attr_ast

def recurse_Attribute(attr, sep='.'):
    if hasattr(attr,'id'):
        return attr.id
    elif isinstance(attr.value,ast.Name):
        return attr.value.id+sep+attr.attr
    elif isinstance(attr.value, ast.Attribute):
        return recurse_Attribute(attr.value)+sep+attr.attr
# Parse a function

# Add nodes and edges to a graph
tmp_count = [0]
def tmp(a):
    a+='_'+str(tmp_count[0])
    tmp_count[0]+=1
    return a

ass_count = [0]
def ass(a):
    a+='_'+str(ass_count[0])
    ass_count[0]+=1
    return a

class EquationNode:
    def __init__(self, label, id=None, node_type:NodeTypes=NodeTypes.VAR, **attrs):
        if not id:
            id = tmp(label)
        self.id = id
        self.label = label

        for k, v in attrs.items():
            setattr(self, k, v)

    def __str__(self):
        return self.id

    def __eq__(self, other):
        return self.id == other.id

class EquationEdge:
    def __init__(self, start: str = None, end: str = None):

        self.start = start
        self.end = end

    def set_start(self, node_id: str):
        self.start = node_id

    def set_end(self, node_id: str):
        self.end = node_id


def parse_(ao, g: Graph, prefix='_', parent: EquationEdge=None):
    # print(ao)
    en=None

    if isinstance(ao, ast.Module):
        for b in ao.body:

            # Check if function def
            if isinstance(b, ast.FunctionDef):
                # Get name of function


                # Parse function
                for b_ in b.body:

                    parse_(b_, g, prefix)

    elif isinstance(ao, ast.Assign):

        assert len(ao.targets) == 1, 'Can only parse assignments with one target'

        target = ao.targets[0]

        # Check if attribute
        if isinstance(ao.targets[0], ast.Attribute) or isinstance(ao.targets[0], ast.Name):

            att = recurse_Attribute(ao.targets[0])
            # print(att)
            target_id = att

        else:
            raise AttributeError('Unknown type of target: ', type(ao.targets[0]))

        target_edge = EquationEdge()
        value_edge = EquationEdge()

        en = EquationNode(target=target_edge, source=value_edge, label='=', ast_type=ast.Assign, node_type=NodeTypes.ASSIGN)
        target_edge.start = en.id
        value_edge.end = en.id




        g.add_edge(target_edge, ignore_missing_nodes=True)
        g.add_edge(value_edge, ignore_missing_nodes=True)

        parse_(ao.value, g, prefix, parent=value_edge.set_start)
        parse_(ao.targets[0], g, prefix, parent=target_edge.set_end)

    elif isinstance(ao, ast.Num):
        # Constant
        #source_var = Variable('c' + str(ao.value), Variable.CONSTANT, val=ao.value)
        source_id = 'c' + str(ao.value)
        en = EquationNode(id=source_id, value = ao.value, node_type=NodeTypes.VAR)


        # Check if simple name
    elif isinstance(ao, ast.Name) or isinstance(ao, ast.Attribute):
        local_id = recurse_Attribute(ao)
        source_id = recurse_Attribute(ao)

        en = EquationNode(id=source_id, local_id=local_id, ast_type=type(ao), label=source_id, node_type=NodeTypes.VAR)


    elif isinstance(ao, ast.UnaryOp):
        # Unary op
        op_sym = get_op_sym(ao.op)
        operand_edge = EquationEdge()
        en = EquationNode(label = ''+op_sym, operand=operand_edge, ast_type=ast.UnaryOp, node_type=NodeTypes.OP)
        operand_edge.end = en.id

        g.add_edge(operand_edge, ignore_missing_nodes=True)

        parse_(ao.operand, g, prefix, parent=operand_edge.set_start)

    elif isinstance(ao, ast.Call):


        op_name = recurse_Attribute(ao.func, sep='.')

        en = EquationNode(label=''+op_name, func=ao.func, args=[], ast_type=ast.Call, node_type=NodeTypes.OP)



        for i, sa in enumerate(ao.args):
            edge_i = EquationEdge(end=en.id)
            g.add_edge(edge_i, ignore_missing_nodes=True)
            parse_(ao.args[i], g, prefix=prefix, parent=edge_i.set_start)


    elif isinstance(ao, ast.BinOp):

        op_sym = get_op_sym(ao.op) # astor.get_op_symbol(ao.op)
        en = EquationNode(label=''+op_sym, left=None, right=None, ast_type=ast.BinOp, node_type=NodeTypes.OP)

        for a in ['left', 'right']:

            operand_edge = EquationEdge(end=en.id)

            g.add_edge(operand_edge, ignore_missing_nodes=True)
            setattr(en,a,operand_edge)
            parse_(getattr(ao, a), g, prefix, parent=operand_edge.set_start)



    else:
        raise TypeError('Cannot parse <' + str(type(ao)) + '>')

    if en:
        g.add_node(en, ignore_exist=True)

    if parent:

        parent(en.id)

def qualify_equation(prefix, g):
    g_qual = Graph()
    g_qual.set_node_map({prefix +'_'+n.id: n for nid, n in g.nodes_map.items()})

    for e in g.edges:
        g_qual.add_edge(EquationEdge(start=prefix +'_'+e.start, end=prefix +'_'+e.end))

    return g_qual

parsed_eq = {}

def parse_eq(scope_id, item, global_graph):
    print(item)
    for eq in item[0]:
        print('name: ', eq.__name__)
        #dont now how Kosher this is: https://stackoverflow.com/questions/20059011/check-if-two-python-functions-are-equal
        eq_key = eq.__qualname__
        print(eq_key)
        if not eq_key in parsed_eq:

            source = inspect.getsource(eq)
            print(source)
            ast_tree = ast.parse(dedent(source))
            g = Graph()
            parse_(ast_tree, g)
            g.as_graphviz()
            print('n nodes local: ', len(g.nodes))
            parsed_eq[eq_key] = (eq, source, g)
        else:
            print('skip parsing')

        g = parsed_eq[eq_key][2]
        g_qualified = qualify_equation(scope_id, g)
        for n in g_qualified.nodes_map.keys():
            print(n)
        global_graph.update(g_qualified)





