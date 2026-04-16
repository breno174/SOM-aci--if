import numpy as np
from helper.node import Node

class SOM:
    """
    Self-Organizing Map (SOM) usando uma estrutura de Nós e Neurônios interligados.
    """
    def __init__(self, line, column, dim, learning_rate=0.5, sigma=None, initial_weights=None):
        self.line = line                  # Número de linhas da grade (altura)
        self.column = column              # Número de colunas da grade (largura)
        self.dim = dim                    # Dimensão dos dados de entrada (número de features)
        self.learning_rate = learning_rate # Taxa de aprendizado inicial para atualização dos pesos
        self.sigma = sigma if sigma is not None else max(line, column) / 2.0 # Raio de vizinhança inicial (vizinhança gaussiana)
        
        # 1. Cria a grade de nós plana
        self.nodes = []
        # Mantém uma matriz 2D de referência puramente lógica para facilitar as ligações
        self.node_grid = [[None for _ in range(column)] for _ in range(line)]

        for i in range(line):
            for j in range(column):
                new_node = Node(row=i, col=j, dim=dim, initial_weights=initial_weights)
                self.nodes.append(new_node)
                self.node_grid[i][j] = new_node

        # 2. Liga os nós uns aos outros (definindo vizinhos diretos)
        self._link_nodes()
        
        # Históricos para plotagem
        self.history = []        # snapshot dos pesos por época
        self.history_lr = []     # taxa de aprendizado por época
        self.history_mse = []    # erro quadrático médio por época

    def _link_nodes(self):
        """
        Interliga os nós para que cada nó conheça seus vizinhos imediatos
        na grade bidimensional (Cima, Baixo, Esquerda, Direita).
        """
        for i in range(self.line):
            for j in range(self.column):
                current_node = self.node_grid[i][j]
                
                # Ligações (Evitando ultrapassar o limite da grade)
                if i > 0:           # Acima
                    current_node.add_neighbor(self.node_grid[i-1][j])
                if i < self.line - 1:  # Abaixo
                    current_node.add_neighbor(self.node_grid[i+1][j])
                if j > 0:           # Esquerda
                    current_node.add_neighbor(self.node_grid[i][j-1])
                if j < self.column - 1:  # Direita
                    current_node.add_neighbor(self.node_grid[i][j+1])

    def find_bmu(self, x):
        """
        Encontra o Node ganhador (Best Matching Unit) iterando sobre todos os nós.
        """
        bmu = None
        min_dist = float('inf')
        
        for node in self.nodes:
            # Distância Euclidiana entre o input x e os pesos do neurônio do nó
            dist = np.linalg.norm(node.neuron.weights - x)
            if dist < min_dist:
                min_dist = dist
                bmu = node
                
        return bmu

    def _decay(self, initial_value, iteration, max_iterations):
        return initial_value * np.exp(-iteration / max_iterations)

    def _compute_mse(self, data):
        """
        Calcula o Erro Quadrático Médio (EQM) da rede para um conjunto de dados.
        Para cada amostra, encontra o BMU e calcula a distância ao quadrado.
        A média dessas distâncias é o EQM (também chamado de Quantization Error).
        """
        total_sq_error = 0.0
        for x in data:
            bmu = self.find_bmu(x)
            total_sq_error += np.sum((x - bmu.neuron.weights) ** 2)
        return total_sq_error / len(data)

    def train(self, data, num_epochs):
        num_samples = data.shape[0]
        max_iterations = num_epochs * num_samples
        
        for epoch in range(num_epochs):
            np.random.shuffle(data)
            
            # Taxa de aprendizado no início desta época
            iter_start = epoch * num_samples
            epoch_lr = self._decay(self.learning_rate, iter_start, max_iterations)
            
            for i, x in enumerate(data):
                iteration = iter_start + i
                
                # 1. Encontrar o BMU
                bmu = self.find_bmu(x)
                
                # 2. Decaimento da taxa de aprendizado e vizinhança
                curr_lr = self._decay(self.learning_rate, iteration, max_iterations)
                curr_sigma = self._decay(self.sigma, iteration, max_iterations)
                
                # 3. Atualizar pesos de toda a rede baseada neste BMU (Cooperação + Adaptação)
                self._update_weights(x, bmu, curr_lr, curr_sigma)
            
            # --- Métricas ao fim da época ---
            # Snapshot dos pesos
            snapshot = np.array([node.neuron.weights.copy() for node in self.nodes])
            self.history.append(snapshot)
            
            # Taxa de aprendizado e EQM
            self.history_lr.append(epoch_lr)
            self.history_mse.append(self._compute_mse(data))

    def _update_weights(self, x, bmu, lr, sigma):
        """
        Atualiza os pesos do BMU e de todos os nós vizinhos usando uma função gaussiana.
        """
        for node in self.nodes:
            # Pegando a distância ao quadrado do nó atual para o BMU na grade
            dist_sq = node.get_distance_to(bmu)
            
            # Cálculo da função de vizinhança (Gaussiana)
            influence = np.exp(-dist_sq / (2 * (sigma**2)))
            
            # Atualiza o neurônio dentro do nó
            node.neuron.update_weights(x, influence, lr)

    def get_weights_matrix(self):
        """
        Retorna todos os pesos em uma matriz formatada (m, n, dim)
        Útil para visualização no Matplotlib.
        """
        weights = np.zeros((self.line, self.column, self.dim))
        for node in self.nodes:
            weights[node.row, node.col] = node.neuron.weights
        return weights
