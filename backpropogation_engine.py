import math 

class value : 
    def __init__(self,data,_children=(),_op= '' , label = ''):
        self.data = data 
        self.prev = set(_children) 
        self.op = _op 
        self.label = label 
        self.grad = 0.0
        self._backward = lambda : None

    def __repr__(self):
        return f"value(data ={self.data:.4f})" 
    
    def __add__(self,other):
        other = other if isinstance(other,value) else value(other)
        out = value(self.data + other.data , (self , other),'+')
        def _backward() : 
            self.grad += 1.0*out.grad 
            other.grad +=1.0*out.grad 
        out._backward = _backward
        return out
    
    def __mul__(self,other) : 
        other = other if isinstance(other,value) else value(other)
        out = value(self.data*other.data , (self,other),'*')
        def _backward(): 
            self.grad += other.data * out.grad 
            other.grad += self.data * out.grad 
        out._backward = _backward
        return out
    
    def __pow__(self,other) : 
        other = other if isinstance(other, value) else value(other)
        out = value(self.data ** other.data, (self, other), f'**{other}')

        def _backward (): 
            self.grad += other.data*(self.data**(other.data-1))*out.grad 

            if self.data > 0 : 
                other.grad += out.grad*((self.data**other.data)*math.log(self.data))
        out._backward = _backward 

        return out 
    
    def exp(self): 
        expo = math.exp(self.data) 
        out = value(expo, (self, ), 'exp')
        def _backward() : 
            self.grad += out.grad * out.data
        out._backward = _backward 
        return out
    
    def tanh(self): 
        expo = math.exp(self.data)
        num = expo**2 - 1 
        div = expo**2 +1 
        answer =  (num /div)
        out = value(answer , (self,),'tanh')
        def _backward() : 
            self.grad += (1 - ( answer **2 ) ) * out.grad
        out._backward = _backward
        return out
    
    
    def back_propogation(self) : 
        topo = [] 
        visited = set() 

        self.grad = 1.0 

        def build_topological_sort (v) : 
            if v not in visited : 
                visited.add(v) 
                for child in v.prev: 
                    build_topological_sort(child) 
                topo.append(v)
        build_topological_sort(self) 

        for node in reversed(topo) : 
            node._backward() 
    
    def leaky_relu(self,alpha = 0.01 ) : 
        out =  value(self.data*alpha if self.data < 0 else self.data , (self , ) , 'leaky_ReLU' ) 

        def _backward() :
            self.grad += (1.0*out.grad if out.data>0 else alpha ) * out.grad 
        out._backward = _backward 
        return out 

    def __radd__(self,other) : 
        return self+other
    
    def __neg__(self) : 
        return self*-1.0 
    
    def __sub__(self,other): 
        return self + (-other)
    
    def __rsub__(self,other) : 
        return other + (-self)
    
    def __rmul__(self,other) : 
        return self*other 
    
    def __truediv__(self, other):
        return self * (other**-1.0)
    
    def __rtruediv__(self,other) : 
        return other * (self**-1.0)