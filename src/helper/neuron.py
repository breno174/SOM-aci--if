import numpy as np

class Neuron:
    """
    Representa um único neurônio na rede, armazenando seus pesos.
    """
    def __init__(self, dim, initial_weights=None):
        if initial_weights is not None:
            self.weights = np.array(initial_weights, dtype=float).copy()
        else:
            self.weights = np.random.random(dim)
        
    def update_weights(self, x, influence, learning_rate):
        """
        Atualiza os pesos do neurônio de acordo com a regra de aprendizado da SOM.
        :param x: Vetor de entrada atual.
        :param influence: Fator de vizinhança calculado para este neurônio.
        :param learning_rate: Taxa de aprendizado atual.
        """
        self.weights += learning_rate * influence * (x - self.weights)
