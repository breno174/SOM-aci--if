import numpy as np
import json
import os
from helper.node import Node

class SOM:
    """
    Self-Organizing Map (SOM) — implementação fiel ao algoritmo de Kohonen.

    A vizinhança é tratada como um campo contínuo de influência (kernel gaussiano
    sobre o grid), sem propagação em cadeia entre neurônios.

    Para cada amostra x apresentada na iteração t:
      1. Encontra o BMU (Best Matching Unit): o neurônio cujo vetor de pesos
         é mais próximo de x no espaço de features.
      2. Atualiza TODOS os neurônios do grid simultaneamente:

         w_i(t+1) = w_i(t) + α(t) · h(bmu, i, t) · (x − w_i(t))

         onde a função de vizinhança gaussiana é:

         h(bmu, i, t) = exp( −‖r_bmu − r_i‖² / (2 · σ(t)²) )

         - ‖r_bmu − r_i‖² é a distância Euclidiana ao quadrado entre as
           posições no GRID (não no espaço de features)
         - σ(t) = σ₀ · exp(−t / T)   (raio de vizinhança com decaimento)
         - α(t) = α₀ · exp(−t / T)   (taxa de aprendizado com decaimento)

    A implementação usa operações vetorizadas com NumPy para eficiência.
    """
    def __init__(self, line, column, dim, learning_rate=0.5, sigma=None, initial_weights=None):
        self.line = line                  # Número de linhas da grade (altura)
        self.column = column              # Número de colunas da grade (largura)
        self.dim = dim                    # Dimensão dos dados de entrada (número de features)
        self.learning_rate = learning_rate # Taxa de aprendizado inicial para atualização dos pesos
        self.sigma = sigma if sigma is not None else max(line, column) / 2.0 # Raio de vizinhança inicial (vizinhança gaussiana)
        
        self.num_neurons = line * column  # Número total de neurônios no grid

        # ── Posições fixas dos neurônios no grid (para cálculo de vizinhança) ──
        # Matriz (num_neurons, 2) com as coordenadas (row, col) de cada neurônio
        rows_idx, cols_idx = np.meshgrid(np.arange(line), np.arange(column), indexing='ij')
        self.grid_positions = np.column_stack([rows_idx.ravel(), cols_idx.ravel()])  # shape: (num_neurons, 2)

        # ── Matriz de pesos dos neurônios (num_neurons, dim) ──
        if initial_weights is not None:
            init_w = np.array(initial_weights, dtype=float)
            self.weights = np.tile(init_w, (self.num_neurons, 1))  # shape: (num_neurons, dim)
        else:
            self.weights = np.random.random((self.num_neurons, dim))

        # ── Estrutura de Nós (mantida para compatibilidade com plotagem) ──
        # Os nós referenciam fatias da matriz self.weights para manter sincronização
        self.nodes = []
        self.node_grid = [[None for _ in range(column)] for _ in range(line)]

        for i in range(line):
            for j in range(column):
                flat_idx = i * column + j
                new_node = Node(row=i, col=j, dim=dim, initial_weights=initial_weights)
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

    def find_bmu(self, x):
        """
        Encontra o BMU (Best Matching Unit) — o neurônio cujo vetor de pesos
        é mais próximo do vetor de entrada x no espaço de features.

        Utiliza operação vetorizada: calcula a distância Euclidiana de x
        a TODOS os neurônios simultaneamente e retorna o índice do menor.

        Returns:
            bmu_idx (int): índice linear do neurônio vencedor em self.weights.
        """
        # Distância Euclidiana de x a cada neurônio — vetorizado
        diffs = self.weights - x                              # (num_neurons, dim)
        dists = np.linalg.norm(diffs, axis=1)                 # (num_neurons,)
        bmu_idx = np.argmin(dists)
        return bmu_idx

    def _decay(self, initial_value, iteration, max_iterations):
        """
        Calcula o decaimento exponencial de um valor ao longo das iterações.
        Equação:
            V(t) = V0 * exp(-t / T)
        Onde:
            V(t) - valor na iteração t
            V0   - valor inicial
            t    - iteração atual
            T    - máximo de iterações
        """
        return initial_value * np.exp(-iteration / max_iterations)

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

    def train(self, data, num_epochs, labeled_data=None):
        """
        Treina a SOM por num_epochs épocas.
        Args:
            data: array de dados de entrada (sem labels).
            num_epochs: número de épocas de treinamento.
            labeled_data: tupla (X, y) com dados e seus verdadeiros labels para
                          calcular a acurácia ao final de cada época. Opcional.
        """
        num_samples = data.shape[0]
        max_iterations = num_epochs * num_samples
        
        for epoch in range(num_epochs):
            np.random.shuffle(data)
            
            # Taxa de aprendizado no início desta época
            iter_start = epoch * num_samples
            epoch_lr = self._decay(self.learning_rate, iter_start, max_iterations)
            epoch_sigma = self._decay(self.sigma, iter_start, max_iterations)
            
            for i, x in enumerate(data):
                iteration = iter_start + i
                
                # 1. Encontrar o BMU (retorna índice linear)
                bmu_idx = self.find_bmu(x)
                
                # 2. Decaimento da taxa de aprendizado e vizinhança
                curr_lr = self._decay(self.learning_rate, iteration, max_iterations)
                curr_sigma = self._decay(self.sigma, iteration, max_iterations)
                
                # 3. Atualizar TODOS os pesos simultaneamente (campo gaussiano)
                self._update_weights(x, bmu_idx, curr_lr, curr_sigma)
            
            # --- Métricas ao fim da época ---
            # Snapshot dos pesos (cópia da matriz vetorizada)
            snapshot = self.weights.copy()
            self.history.append(snapshot)
            
            # Taxa de aprendizado e EQM
            self.history_lr.append(epoch_lr)
            self.history_sigma.append(epoch_sigma)
            mse = self._compute_mse(data)
            self.history_mse.append(mse)
            
            # Acurácia por época (opcional)
            if labeled_data is not None:
                X_labeled, y_labeled = labeled_data
                acc = self.compute_accuracy(X_labeled, y_labeled)
                self.history_accuracy.append(acc)
                print(f"  Época {epoch+1}/{num_epochs} | EQM: {mse:.4f} | Acurácia: {acc*100:.1f}%")
            else:
                print(f"  Época {epoch+1}/{num_epochs} | EQM: {mse:.4f}")

    def _update_weights(self, x, bmu_idx, lr, sigma):
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
        bmu_pos = self.grid_positions[bmu_idx]                 # (2,)

        # Distância ao quadrado de TODOS os neurônios ao BMU no grid
        delta_pos = self.grid_positions - bmu_pos              # (num_neurons, 2)
        dist_sq = np.sum(delta_pos ** 2, axis=1)               # (num_neurons,)

        # Função de vizinhança gaussiana — campo contínuo de influência
        influence = np.exp(-dist_sq / (2 * (sigma ** 2)))      # (num_neurons,)

        # Atualização vetorizada: w_i += α · h_i · (x − w_i)
        diff = x - self.weights                                # (num_neurons, dim)
        self.weights += lr * influence[:, np.newaxis] * diff   # broadcasting (num_neurons, dim)

    def get_weights_matrix(self):
        """
        Retorna todos os pesos em uma matriz formatada (m, n, dim)
        Útil para visualização no Matplotlib.
        """
        return self.weights.reshape(self.line, self.column, self.dim)

    def get_cluster_grid(self, cluster_labels):
        """
        Dada uma lista de labels (índice de cluster para cada neurônio),
        retorna uma grade 2D (line x column) com o label de cluster de cada posição.
        A ordem dos labels segue a mesma de self.weights (row-major).
        """
        return np.array(cluster_labels).reshape(self.line, self.column)

    def compute_accuracy(self, data, true_labels):
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

    def label_nodes_by_data(self, data, true_labels):
        """
        Para cada neurônio da grade, associa o label mais frecuente entre as
        amostras de dados cujo BMU é aquele neurônio.
        Retorna uma grade 2D (line x column) com o label dominante em cada
        célula (ou -1 se sem amostras) e um dicionário {(row, col): [labels]}.
        """
        from collections import Counter
        
        # Mapeia cada neurônio (por posição no grid) para uma lista de labels
        node_label_map = {(r, c): [] for r in range(self.line) for c in range(self.column)}
        
        for x, label in zip(data, true_labels):
            bmu_idx = self.find_bmu(x)
            pos = self.grid_positions[bmu_idx]
            node_label_map[(pos[0], pos[1])].append(label)
        
        # Grade com o label dominante
        label_grid = np.full((self.line, self.column), -1, dtype=int)
        for (row, col), labels in node_label_map.items():
            if labels:
                label_grid[row, col] = Counter(labels).most_common(1)[0][0]
        
        return label_grid, node_label_map

    def plot_confusion_matrix(self, data, true_labels, class_names, save_path=None):
        """
        Calcula e plota a matriz de confusão da SOM diretamente.
        Args:
            data: array de dados normalizados de entrada.
            true_labels: array com os labels reais (inteiros a partir de 0).
            class_names: lista com os nomes das classes, na ordem dos inteiros.
            save_path: caminho opcional para salvar a imagem (ex: 'src/confusion.png').
        Returns:
            matrix: a matriz de confusão calculada (numpy array).
            accuracy: acurácia global calculada.
        """
        import matplotlib.pyplot as plt

        num_classes = len(class_names)
        matrix = self.compute_confusion_matrix(data, true_labels, num_classes)
        accuracy = self.compute_accuracy(data, true_labels)

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(matrix, cmap='Blues')

        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels(class_names, rotation=30, ha='right')
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predito (SOM/BMU)", fontweight='bold')
        ax.set_ylabel("Real", fontweight='bold')
        ax.set_title(
            f"Matriz de Confusão – SOM\nAcurácia: {accuracy * 100:.1f}%",
            fontweight='bold'
        )

        for row in range(num_classes):
            for col in range(num_classes):
                val = matrix[row, col]
                color = 'white' if val > matrix.max() / 2 else 'black'
                ax.text(col, row, str(val), ha='center', va='center',
                        fontsize=14, color=color, fontweight='bold')

        plt.colorbar(im, ax=ax)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
            print(f"Matriz de confusão salva em: {save_path}")

        plt.show()
        return matrix, accuracy

    def plot_mse_history(self, save_path=None):
        """
        Plota o Erro Quadrático Médio (EQM) registrado em self.history_mse.
        Cada ponto corresponde a uma época (ou intervalo amostral registrado manualmente).
        """
        import matplotlib.pyplot as plt
        if not self.history_mse:
            print("Nenhum histórico de EQM disponível para plotar.")
            return

        x = list(range(1, len(self.history_mse) + 1))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, self.history_mse, color='crimson', linewidth=2.5,
                marker='o', markersize=5, label='EQM')
        ax.fill_between(x, self.history_mse, alpha=0.15, color='crimson')
        ax.set_title("Erro Quadrático Médio (EQM) por Época", fontsize=13, fontweight='bold')
        ax.set_xlabel("Época", fontsize=11)
        ax.set_ylabel("EQM médio", fontsize=11)
        ax.set_xticks(x)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
            print(f"Gráfico de EQM salvo em: {save_path}")
        plt.show()

    def plot_accuracy_history(self, class_names=None, save_path=None):
        """
        Plota a acurácia registrada em self.history_accuracy por época.
        Requer que o treinamento tenha sido executado com labeled_data
        ou que history_accuracy tenha sido populado manualmente.
        """
        import matplotlib.pyplot as plt
        if not self.history_accuracy:
            print("Nenhum histórico de acurácia disponível para plotar.")
            return

        x = list(range(1, len(self.history_accuracy) + 1))
        acc_pct = [a * 100 for a in self.history_accuracy]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, acc_pct, color='steelblue', linewidth=2.5,
                marker='s', markersize=5, label='Acurácia (%)')
        ax.fill_between(x, acc_pct, alpha=0.15, color='steelblue')
        ax.set_title("Acurácia por Época", fontsize=13, fontweight='bold')
        ax.set_xlabel("Época", fontsize=11)
        ax.set_ylabel("Acurácia (%)", fontsize=11)
        ax.set_ylim(0, 105)
        ax.set_xticks(x)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
            print(f"Gráfico de acurácia salvo em: {save_path}")
        plt.show()

    def plot_neighborhood_decay(self, max_iterations=None, save_path=None):
        """
        Plota o decaimento exponencial do raio de vizinhança (sigma) e da taxa de aprendizado.
        Traça por época se os históricos estiverem disponíveis, senão volta a traçar a 
        linha teórica pelas iterações.
        """
        import matplotlib.pyplot as plt
        
        has_history = len(self.history_sigma) > 0 and len(self.history_lr) > 0

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        
        if has_history:
            fig.suptitle("Decaimento dos Parâmetros da SOM ao Longo das Épocas",
                         fontsize=13, fontweight='bold')
            x_vals = list(range(1, len(self.history_sigma) + 1))
            sigma_vals = self.history_sigma
            lr_vals = self.history_lr
            x_label = "Época"
            marker = 'o'
        else:
            fig.suptitle("Decaimento dos Parâmetros da SOM ao Longo das Iterações",
                         fontsize=13, fontweight='bold')
            if max_iterations is None:
                print("Aviso: sem histórico e max_iterations não providenciado.")
                return
            x_vals = np.arange(0, max_iterations + 1)
            sigma_vals = self.sigma * np.exp(-x_vals / max_iterations)
            lr_vals    = self.learning_rate * np.exp(-x_vals / max_iterations)
            x_label = "Iteração"
            marker = None

        # Sigma
        axes[0].plot(x_vals, sigma_vals, color='darkorange', linewidth=2.2, marker=marker, markersize=5)
        axes[0].fill_between(x_vals, sigma_vals, alpha=0.15, color='darkorange')
        axes[0].set_title("Decaimento do Raio de Vizinhança (σ)", fontweight='bold')
        axes[0].set_xlabel(x_label)
        axes[0].set_ylabel("σ (sigma)")
        if has_history and len(x_vals) <= 20: 
            axes[0].set_xticks(x_vals)
        axes[0].grid(True, linestyle='--', alpha=0.4)

        # Learning rate
        axes[1].plot(x_vals, lr_vals, color='purple', linewidth=2.2, marker=marker, markersize=5)
        axes[1].fill_between(x_vals, lr_vals, alpha=0.15, color='purple')
        axes[1].set_title("Decaimento da Taxa de Aprendizado (α)", fontweight='bold')
        axes[1].set_xlabel(x_label)
        axes[1].set_ylabel("α (learning rate)")
        if has_history and len(x_vals) <= 20: 
            axes[1].set_xticks(x_vals)
        axes[1].grid(True, linestyle='--', alpha=0.4)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
            print(f"Gráfico de decaimento salvo em: {save_path}")
        plt.show()

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
