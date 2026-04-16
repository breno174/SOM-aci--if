from .neuron import Neuron

class Node:
    """
    Representa um nó na grade da SOM. Cada nó contém um Neurônio e
    tem conhecimento de suas coordenadas e nós vizinhos diretos.
    """
    def __init__(self, row, col, dim, initial_weights=None):
        self.row = row
        self.col = col
        self.neuron = Neuron(dim, initial_weights)
        
        # Lista de referências para nós vizinhos (cima, baixo, esquerda, direita)
        self.neighbors = []
        
    def add_neighbor(self, neighbor_node):
        """
        Adiciona um nó vizinho à lista de conexões deste nó.
        """
        if neighbor_node not in self.neighbors:
            self.neighbors.append(neighbor_node)

    def get_distance_to(self, other_node):
        """
        Calcula a distância ao quadrado na grade até outro nó.
        Pode ser usada para a função de vizinhança.
        """
        return (self.row - other_node.row)**2 + (self.col - other_node.col)**2
