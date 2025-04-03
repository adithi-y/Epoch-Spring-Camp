import numpy as np

# first you define a class called node and give it the following attributes
# value: which is the value that node outputs after performing the computation
# children: the children that node creates
# operation: the operation that node performs
# operation constant: any constant arguments for the operation if needed

class node:
    def __init__(self, value, children=[], op=None, opconst=None):
        self.value = np.array(value)
        self.children = children
        self.op = op
        self.opconst = opconst
        self.grad = None

# now you define two functions for outputting reader friendly values rather than hexadecimal representations of the outputs

    def __repr__(self):
        return f"Node(value={self.value}, op={self.op})"

    def __str__(self):
        return self.__repr__()
    
# after this you define functions for adding subtracting multiplying dividing and exponentiating any two nodes
# you name the functions with the underscores so python knows to call these functions when encountering a symbol like +,-,*,/,**
# in each of these functions you take a case of performing it for two nodes and the case of performing it for a node and a constant

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
        

# then you define reverse functions in case you have computations like 2-node or 5/node

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
    
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.value)
        for child in self.children:
            if self.op == 'add':
                child.grad+=self.grad

            elif self.op == 'sub':
                if child == self.children[0]:
                    child.grad+=self.grad
                else:
                    child.grad-=self.grad
            elif self.op == 'mul':
                if len(self.children) == 2:
                    if child is self.children[0]:
                        other = self.children[1]  
                    else:
                        other = self.children[0]
                    child.grad += self.grad*other.value
                else:
                    const = self.opconst if opconst is not None else 1
                    child.grad = self.grad*const
            elif self.op == 'div':
                if len(self.children) == 2:
                    numerator, denominator = self.children
                    if child is numerator:
                        child.grad += self.grad / denominator.value
                    else:
                        child.grad += -self.grad * numerator.value / (denominator.value ** 2)
                else:
                    if 'numerator' in self.op_args:  # Constant numerator case
                        child.grad += -self.grad * self.op_args['numerator'] / (child.value ** 2)
                    else:  # Constant denominator case
                        child.grad += self.grad / self.op_args.get('denominator', 1)
            elif self.op == 'pow':
                exponent = self.op_args['exponent']
                child.grad += self.grad * exponent * (child.value ** (exponent - 1))

# then you make objects a,b and c and for the value of these objects you input numpy arrays

a = node(np.array([1,5,10]))
b = node(np.array([2,4,6]))
c = node(np.array([1,3,5]))

# then you assign the value of the computation a*b+c to another node d

d = a*b+c

# and finally you print the value of d(an array), the children of d(and all their attributes), and the operation of d

print(d.value)
print(d.children)
print(d.op)
print(d.backward())