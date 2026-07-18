from graphviz import Digraph 

def trace(root) : 
    # builds a set of all nodes and edges in a graph 
    nodes , edges = set() , set() 
    def build(v) : 
        if v not in nodes : 
            nodes.add(v) 
            for child in v.prev : 
                edges.add((child,v)) 
                build(child) 
    build(root) 
    return nodes , edges 

def draw_connections(root) : 
    connections = Digraph( format='svg' , graph_attr={'rankdir':'LR'})

    nodes , edges = trace(root) 
    for n in nodes : 
        UID = str(id(n))
        connections.node(name = UID, label = " { %s | data%.4f | grad= %.4f } " %(n.label , n.data , n.grad) , shape = 'record' )

        if n.op : 
            connections.node(name = UID + n.op , label = n.op )
            connections.edge(UID + n.op , UID )
    for n1 , n2 in edges : 
        connections.edge(str(id(n1)), str(id(n2)) + n2.op)

    return connections 