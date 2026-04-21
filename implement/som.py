import numpy as np
import math
import random
from src.helper.node import Node
from src.helper.neuron import Neuron

class SOM:
    def __init__(self, initial_weights, line, column, dimension, learning_rate=0.1, sigma=None):
        self.line = line
        self.column = column
        self.dimension = dimension
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.sigma = sigma if sigma is not None else max(line, column) / 2.0 # Raio de vizinhança inicial (vizinhança gaussiana)
        self.init_sigma = sigma if sigma is not None else max(line, column) / 2.0 # Raio de vizinhança inicial (vizinhança gaussiana)
        self.num_neurons = line * column
        
        # ── Matriz de pesos dos neurônios (num_neurons, dim) ──
        if initial_weights is not None:
            init_w = np.array(initial_weights, dtype=float)
            self.weights = np.tile(init_w, (self.num_neurons, 1))
    
        # ── Posições fixas dos neurônios no grid (para cálculo de vizinhança) ──
        # Matriz (num_neurons, 2) com as coordenadas (row, col) de cada neurônio
        rows_idx, cols_idx = np.meshgrid(np.arange(line), np.arange(column), indexing='ij')
        self.grid_positions = np.column_stack([rows_idx.ravel(), cols_idx.ravel()])  # shape: (num_neurons, 2)
        self.node_grid = [[None for _ in range(column)] for _ in range(line)]    
        self.nodes = []

        for i in range(line):
            for j in range(column):
                flat_idx = i * column + j
                new_node = Node(row=i, col=j, dim=self.dimension, initial_weights=initial_weights)
                # Sincroniza: o neurônio do Node aponta para a mesma fatia da matriz
                new_node.neuron.weights = self.weights[flat_idx]
                self.nodes.append(new_node)
                self.node_grid[i][j] = new_node
        
        # Liga vizinhos (usado apenas para desenhar linhas de topologia nos gráficos)
        self._link_nodes()

        # Históricos para plotagem
        self.history = []           # snapshot dos pesos por época
        self.history_lr = []        # taxa de aprendizado por época
        self.history_mse = []       # erro quadrático médio por época
        self.history_accuracy = []  # acurácia por época (só preenchido se labeled_data for passado ao train)
        self.history_sigma = []     # raio de vizinhança por época

    def _link_nodes(self):
        """
        Interliga os nós para que cada nó conheça seus vizinhos imediatos
        na grade bidimensional (Cima, Baixo, Esquerda, Direita).
        Nota: usado APENAS para visualização da topologia nos gráficos.
        A atualização de pesos NÃO usa esta estrutura de vizinhos.
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

    def _euclidean_distance(self, x, w):
        soma = 0
        for a, b in zip(x, w):
            diferenca = a - b
            soma += diferenca**2
        return math.sqrt(soma)

    def _winner(self, x):
        """Retorna o índice do neurônio vencedor.
        Em caso de empate (distâncias iguais), sorteia aleatoriamente entre os candidatos.
        Isso evita que apenas o neurônio 0 vença quando todos partem do mesmo ponto.
        """
        distancias = [self._euclidean_distance(x, w) for w in self.weights]
        min_distance = min(distancias)

        # Coleta todos os índices com a mesma distância mínima (com tolerância numérica)
        candidatos = [
            i for i, d in enumerate(distancias) if abs(d - min_distance) < 1e-9
        ]
        return random.choice(candidatos)
    
    def find_bmu(self, x):
        """
        Encontra o BMU (Best Matching Unit) — o neurônio cujo vetor de pesos
        é mais próximo do vetor de entrada x no espaço de features.

        Utiliza operação vetorizada: calcula a distância Euclidiana de x
        a TODOS os neurônios simultaneamente e retorna o índice do menor.

        Returns:
            bmu_idx (int): índice linear do neurônio vencedor em self.weights.
        """
        bmu_idx = None
        min_dist = float('inf')
        # percorre todos os neurônios
        for i, w in enumerate(self.weights):
            # aproveita a função da distância euclidiana
            dist = self._euclidean_distance(x, w)
            # verifica se é o menor até agora
            if dist < min_dist:
                min_dist = dist
                bmu_idx = i
        return bmu_idx

    def _decay(self, interaction, max_interactions):
        return self.initial_learning_rate * np.exp(-interaction/max_interactions) 

    def neighborhood_decay(self, interaction, max_interactions):
        # return np.exp(-interaction / max_interactions)
        return self.init_sigma * ((1 / self.init_sigma)**(interaction/max_interactions))

    def _update_weights(self, x, bmu_inx, lr, sigma):
        """
        Atualiza os pesos de TODOS os neurônios do grid simultaneamente,
        usando uma função de vizinhança gaussiana centrada no BMU.

        Não há propagação em cadeia (BFS/DFS). A influência de cada neurônio
        é calculada diretamente a partir da sua distância ao BMU no grid:

            h(bmu, i) = exp( −‖r_bmu − r_i‖² / (2 · σ²) )
            w_i ← w_i + α · h(bmu, i) · (x − w_i)

        A operação é completamente vetorizada com NumPy.

        Args:
            x:       vetor de entrada (dim,)
            bmu_idx: índice linear do BMU em self.weights
            lr:      taxa de aprendizado atual α(t)
            sigma:   raio de vizinhança atual σ(t)
        """
        # Posição do BMU no grid
        pos_bmu = self.grid_positions[bmu_inx]
        delta = self.grid_positions - pos_bmu
        dist_sq = np.sum(delta**2, axis=1)
        
        # h(bmu, i) = exp( -dist_sq / (2 * sigma^2) )
        h = np.exp(-dist_sq / (2 * (sigma**2)))
        
        # w_i = w_i + lr * h_i * (x - w_i)
        # self.weights shape: (num_neurons, dimension)
        # h shape: (num_neurons,) -> (num_neurons, 1) para broadcast
        self.weights += lr * h[:, np.newaxis] * (x - self.weights)

    def _compute_mse(self, data):
        """
        Calcula o Erro Quadrático Médio (EQM) da rede para um conjunto de dados.
        Para cada amostra, encontra o BMU e calcula a distância ao quadrado.
        A média dessas distâncias é o EQM (também chamado de Quantization Error).
        """
        total_sq_error = 0.0
        for x in data:
            bmu_idx = self.find_bmu(x)
            total_sq_error += np.sum((x - self.weights[bmu_idx]) ** 2)
        return total_sq_error / len(data)

    def quantization_error(self, data):
        """Alias público para _compute_mse (Quantization Error)."""
        return self._compute_mse(data)

    def _compute_accuracy(self, data, true_labels):
        """
        Calcula a acurácia da SOM como um classificador.
        Para cada neurônio, determina o label dominante entre as amostras que o elegeram como BMU.
        A acurácia é a fração de amostras corretamente classificadas pelo label do seu BMU.
        """
        from collections import Counter
        
        # Passo 1: Descobrir o label dominante de cada neurônio
        neuron_labels = {idx: [] for idx in range(self.num_neurons)}
        for x, label in zip(data, true_labels):
            bmu_idx = self.find_bmu(x)
            neuron_labels[bmu_idx].append(label)
        
        dominant_label = {}
        for idx, labels in neuron_labels.items():
            if labels:
                dominant_label[idx] = Counter(labels).most_common(1)[0][0]
        
        # Passo 2: Classificar cada amostra pelo label do seu BMU e contar acertos
        correct = 0
        for x, true_label in zip(data, true_labels):
            bmu_idx = self.find_bmu(x)
            pred = dominant_label.get(bmu_idx, -1)
            if pred == true_label:
                correct += 1
        
        return correct / len(true_labels)

    def train(self, data, num_epoch=100, labeled_data=None):
        """
        Treina a SOM por num_epochs épocas.
        Args:
            data: array de dados de entrada (sem labels).
            num_epochs: número de épocas de treinamento.
            labeled_data: tupla (X, y) com dados e seus verdadeiros labels para
                          calcular a acurácia ao final de cada época. Opcional.
        """
        num_sample = data.shape[0]
        max_interactions = num_epoch * num_sample
        print("sigma", self.sigma)

        for epoch in range(num_epoch):
            np.random.shuffle(data)

            # Taxa de aprendizado e sigma no início desta época
            iter_start = epoch * num_sample
            epoch_lr = self.learning_rate = self._decay(iter_start, max_interactions)
            epoch_sigma = self.sigma = self.neighborhood_decay(iter_start, max_interactions)

            for i in range(num_sample):
                interaction = epoch * num_sample + 1
                x = data[i]
                # 1. Encontrar o BMU (retorna índice linear)
                bmu_idx = self.find_bmu(x)
                # 2. Decaimento da taxa de aprendizado e vizinhança
                curr_lr = self._decay(iter_start, max_interactions)
                curr_sigma = self.neighborhood_decay(iter_start, max_interactions)
                
                # 3. Atualizar TODOS os pesos simultaneamente (campo gaussiano)
                self._update_weights(x, bmu_idx, curr_lr, curr_sigma)
            
            # --- Métricas ao fim da época ---
            snapshot = self.weights.copy()
            self.history.append(snapshot)
            # Taxa de aprendizado e EQM
            self.history_lr.append(epoch_lr)
            self.history_sigma.append(epoch_sigma)
            mse = self._compute_mse(data)
            self.history_mse.append(mse)

            if labeled_data is not None:
                X_labeled, y_labeled = labeled_data
                acc = self._compute_accuracy(X_labeled, y_labeled)
                self.history_accuracy.append(acc)
                print(f"Epoca {epoch+1}/{num_epoch} | LR={epoch_lr:.4f} | sigma={epoch_sigma:.4f} | MSE={mse:.4f} | Acc={acc:.2%}")
            else:
                print(f"Epoca {epoch+1}/{num_epoch} | LR={epoch_lr:.4f} | sigma={epoch_sigma:.4f} | MSE={mse:.4f}")
    
    def compute_confusion_matrix(self, data, true_labels, num_classes):
        """
        Gera uma matriz de confusão da SOM como classificador.
        Cada amostra é classificada pelo label dominante do seu BMU.
        Retorna uma matriz (num_classes x num_classes) onde:
            confusion[true][predicted] = contagem
        """
        from collections import Counter
        
        # Descobre o label dominante de cada neurônio
        neuron_labels = {idx: [] for idx in range(self.num_neurons)}
        for x, label in zip(data, true_labels):
            bmu_idx = self.find_bmu(x)
            neuron_labels[bmu_idx].append(label)
        
        dominant_label = {}
        for idx, labels in neuron_labels.items():
            if labels:
                dominant_label[idx] = Counter(labels).most_common(1)[0][0]
        
        # Monta a matriz de confusão
        matrix = np.zeros((num_classes, num_classes), dtype=int)
        for x, true_label in zip(data, true_labels):
            bmu_idx = self.find_bmu(x)
            pred = dominant_label.get(bmu_idx, -1)
            if pred >= 0:
                matrix[true_label][pred] += 1
        
        return matrix


    def get_cluster_grid(self, cluster_labels):
        """
        Dada uma lista de labels (índice de cluster para cada neurônio),
        retorna uma grade 2D (line x column) com o label de cluster de cada posição.
        A ordem dos labels segue a mesma de self.weights (row-major).
        """
        return np.array(cluster_labels).reshape(self.line, self.column)

    def label_nodes_by_data(self, data, true_labels):
        """
        Para cada neurônio da grade, associa o label mais frequente entre as
        amostras de dados cujo BMU é aquele neurônio.
        Retorna uma grade 2D (line x column) com o label dominante em cada
        célula (ou -1 se sem amostras) e um dicionário {(row, col): [labels]}.
        """
        from collections import Counter

        node_label_map = {(r, c): [] for r in range(self.line) for c in range(self.column)}

        for x, label in zip(data, true_labels):
            bmu_idx = self.find_bmu(x)
            pos = self.grid_positions[bmu_idx]
            node_label_map[(pos[0], pos[1])].append(label)

        label_grid = np.full((self.line, self.column), -1, dtype=int)
        for (row, col), labels in node_label_map.items():
            if labels:
                label_grid[row, col] = Counter(labels).most_common(1)[0][0]

        return label_grid, node_label_map

    @staticmethod
    def save_experiment(path, seed, initial_weights, sample_order, config=None):
        """
        Salva a topologia completa do experimento em disco para permitir replicá-lo.
        Args:
            path: caminho do arquivo de saída (.json).
            seed: semente aleatória usada no experimento.
            initial_weights: array (num_neurons x dim) com os pesos iniciais antes do treino.
            sample_order: lista com os índices das amostras na ordem em que foram apresentadas.
            config: dicionário opcional com hiperparâmetros da SOM (linha, coluna, lr, sigma...).
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        experiment = {
            'seed': int(seed),
            'initial_weights': initial_weights.tolist(),
            'sample_order': [int(i) for i in sample_order],
            'config': config or {}
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(experiment, f, indent=2, ensure_ascii=False)
        print(f"Topologia do experimento salva em: {path}")

    @staticmethod
    def load_experiment(path):
        """
        Carrega a topologia de um experimento salvo anteriormente.
        Returns:
            dict com 'seed', 'initial_weights' (np.array), 'sample_order' (list) e 'config' (dict).
        """
        with open(path, 'r', encoding='utf-8') as f:
            experiment = json.load(f)
        experiment['initial_weights'] = np.array(experiment['initial_weights'])
        print(f"Topologia carregada de: {path}")
        return experiment