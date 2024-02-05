import math 
class Value:
    
    
    def __init__(self,data, _parents = (), _op = ''):
        self.data = data  #value
        self.grad = 0     #derivative (slope/gradient)  
        self.backward = lambda : None #fn to propogate
        self.prev = set(_parents)    
        self._op = _op     #operator used for creation
        
    def __repr__(self):
        return f"Value(data={self.data})"
    
    #addition
    def __add__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data + other.data , (self,other), '+')
        
        def _backward():
            self.grad += out.grad               #Setting the gradient when backpropogating
            other.grad += self.grad
        out.backward = _backward    
        
        return out
    
    #multiplication
    def __mul__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data * other.data , (self,other), '*')
        
        def _backward():
            self.grad += out.grad*other.data               #Setting the gradient when backpropogating
            other.grad += self.grad *self.data
        out.backward = _backward  
        
        return out
    
    #raisng to an integer or float power
    def __pow__(self,other):
        assert isinstance(other, (int,float)) #only float or int powers
        out = Value(self**other, (self,) , f'**{other}') #using fstring to specify the power used for graph visualisation 
        
        def _backward():
            self.grad = other*(out.grad**(other-1))
        out.backward = _backward
        
        return out
    
    #the squashing function to limit it to <1
    def tanh(self):
        self = self if isinstance(self,Value) else Value(self)
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')
        
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        
        return out
    
    #backpropagting by using topo sort to go from the result to the parents
    def backward(self):
        
        topo = []
        vis = set()
        
        def build_topo(v):
            if v not in vis:
                vis.add(v)
                for parent in v._prev:
                    build_topo(parent)
                topo.append(v)
        #building the topo sort      
        build_topo(self)
        
        #actually backpropogating
        self.grad = 1
        for v in reversed(topo):
            v._backward()
            
    #defining negative numbers
    def __neg__(self):
        return self*(-1)

    def __radd__(self,other):
        return self + other
    
    def __rmul__(self,other):
        return self*other
    
    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1
        
               
print(Value.tanh(2))     
        