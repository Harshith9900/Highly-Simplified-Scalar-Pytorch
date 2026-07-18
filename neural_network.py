from backpropogation_engine import value 
import random 

class Neuron : 
    def __init__(self,no_inputs): 
        self.w = [value(random.uniform(-1,1)) for _ in range(no_inputs)]
        self.b = value(random.uniform(-1,1))

    def __call__(self,x) : 
        # ALL(weights (w) * inputs (x) ) + bias (b) 
        neuron_result = sum((wi*xi for wi , xi in zip(self.w,x)) , self.b )
        return neuron_result.tanh() 
    
    def parameters(self) : 
        return self.w + [self.b] 

class Layer : 
    def __init__(self,no_inputs,no_outputs) : 
        self.neurons = [Neuron(no_inputs) for _ in range(no_outputs)]
    def __call__(self , x ) : 
        outs = [ n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs 
    
    def parameters(self) : 
        return [ p for neuron in self.neurons for p in neuron.parameters() ]
    
class MLP : 
    def __init__(self,no_inputs , no_outputs ):
        size = [no_inputs] + no_outputs 
        self.layer = [Layer(size[i] , size[i+1]) for i in range(len(no_outputs))]

    def __call__(self,x) : 
        for layer in self.layer : 
            x = layer(x) 
        return x 
    
    def parameters(self) : 
        return [ p for layer in self.layer for p in layer.parameters()]
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0