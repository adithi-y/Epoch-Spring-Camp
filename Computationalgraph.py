import numpy as np

class node:
    def __init__(self, value, children=[], op=None, opconst=None):
        self.value = np.array(value)
        self.children = children
        self.op = op
        self.opconst = opconst

    def __repr__(self):
        return f"Node(value={self.value}, op={self.op})"

    def __str__(self):
        return self.__repr__()


    def __add__(self, other):
        if isinstance(other,node):
            return node(self.value+other.value, [self, other], "add", None)
        else:
            return node(self.value+other, [self], "add", other)
        
    def __sub__(self, other):
        if isinstance(other, node):
            return node(self.value-other.value, [self, other], "sub", None)
        else:
            return node(self.value-other, [self], "sub", other)
        
    def __mul__(self, other):
        if isinstance(other, node):
            return node(self.value*other.value, [self, other], "mul", None)
        else:
            return node(self.value*other, [self], "mul", other)
        
    def __truediv__(self, other):
        if isinstance(other, node):
            return node(self.value/other.value, [self, other], "div", None)
        else:
            return node(self.value/other, [self], "div", other)
        
    def __pow__(self, other):
        if isinstance(other, node):
            return node(self.value**other.value, [self, other], "pow", None)
        else:
            return node(self.value**other, [self], "pow", other)
        





    def __radd__(self, other):
        return node(self.value+other, [self], "add", other)
        
    def __rsub__(self, other):
        return node(self.value-other, [self], "sub", other)
        
    def __rmul__(self, other):
        return node(self.value*other, [self], "mul", other)
        
    def __rtruediv__(self, other):
        return node(self.value/other, [self], "div", other)
        
    def __rpow__(self, other):
        return node(self.value**other, [self], "pow", other)
    

a = node(np.array([1,5,10]))
b = node(np.array([2,4,6]))
c = node(np.array([1,3,5]))
d = a*b+c
print(d.value)
print(d.children)
print(d.op)



# first you define a class called node and give it the following attributes
# value: which is the value that node outputs after performing the computation
# children: the children that node creates
# operation: the operation that node performs
# operation constant: any constant arguments for the operation if needed
# now you define two functions for outputting reader friendly values rather than hexadecimal representations of the outputs
# after this you define functions for adding subtracting multiplying dividing and exponentiating any two nodes
# you name the functions with the underscores so python knows to call these functions when encountering a symbol like +,-,*,/,**
# in each of these functions you take a case of performing it for two nodes and the case of performing it for a node and a constant
# then you define reverse functions in case you have computations like 2-node or 5/node
# then you make objects a,b and c and for the value of these objects you input numpy arrays
# then you assign the value of the computation a*b+c to another node d
# and finally you print the value of d(an array), the children of d(and all their attributes), and the operation of d