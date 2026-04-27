import numpy as np
import math
import random
from src.helper.node import Node
from src.helper.neuron import Neuron
import os
import json


class SOM:
    def __init__(
        self, initial_weights, line, column, dimension, learning_rate=0.1, sigma=None
    ):
        self.line = line
        self.column = column
        self.dimension = dimension
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.sigma = (
            sigma if sigma is not None else max(line, column) / 2.0
        )  # Raio de vizinhança inicial (vizinhança gaussiana)
        self.init_sigma = (
            sigma if sigma is not None else max(line, column) / 2.0
        )  # Raio de vizinhança inicial (vizinhança gaussiana)
        self.num_neurons = line * column

        # ── Matriz de pesos dos neurônios (num_neurons, dim) ──
        if initial_weights is not None:
            init_w = np.array(initial_weights, dtype=float)
            self.weights = np.random.rand(self.num_neurons, self.dimension)

        # ── Posições fixas dos neurônios no grid (para cálculo de vizinhança) ──
        # Matriz (num_neurons, 2) com as coordenadas (row, col) de cada neurônio
        rows_idx, cols_idx = np.meshgrid(
            np.arange(line), np.arange(column), indexing="ij"
        )
        self.grid_positions = np.column_stack(
            [rows_idx.ravel(), cols_idx.ravel()]
        )  # shape: (num_neurons, 2)
        self.node_grid = [[None for _ in range(column)] for _ in range(line)]
        self.nodes = []

        for i in range(line):
            for j in range(column):
                flat_idx = i * column + j
                new_node = Node(
                    row=i, col=j, dim=self.dimension, initial_weights=initial_weights
                )
                # Sincroniza: o neurônio do Node aponta para a mesma fatia da matriz
                new_node.neuron.weights = self.weights[flat_idx]
                self.nodes.append(new_node)
                self.node_grid[i][j] = new_node

        # Liga vizinhos (usado apenas para desenhar linhas de topologia nos gráficos)
        self._link_nodes()

        # Históricos para plotagem
        self.history = []  # snapshot dos pesos por época
        self.history_lr = []  # taxa de aprendizado por época
        self.history_mse = []  # erro quadrático médio por época
        self.history_accuracy = (
            []
        )  # acurácia por época (só preenchido se labeled_data for passado ao train)
        self.history_sigma = []  # raio de vizinhança por época

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
                if i > 0:  # Acima
                    current_node.add_neighbor(self.node_grid[i - 1][j])
                if i < self.line - 1:  # Abaixo
                    current_node.add_neighbor(self.node_grid[i + 1][j])
                if j > 0:  # Esquerda
                    current_node.add_neighbor(self.node_grid[i][j - 1])
                if j < self.column - 1:  # Direita
                    current_node.add_neighbor(self.node_grid[i][j + 1])

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
        min_dist = float("inf")
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
        return self.initial_learning_rate * np.exp(
            -interaction / max_interactions * 0.5
        )

    def neighborhood_decay(self, interaction, max_interactions):
        sigma_final = 0.1
        return self.init_sigma * (
            (sigma_final / self.init_sigma) ** (interaction / max_interactions)
        )

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

    def compute_accuracy(self, X_train, y_train, X_test, y_test):
        """
        Calcula a acurácia da SOM como um classificador.
        Para cada neurônio, determina o label dominante entre as amostras que o elegeram como BMU.
        A acurácia é a fração de amostras corretamente classificadas pelo label do seu BMU.
        """
        from collections import Counter

        # ── 1. rotular neurônios com dados de treino ──
        neuron_labels = {}

        for x, y in zip(X_train, y_train):
            bmu = self.find_bmu(x)
            neuron_labels.setdefault(bmu, []).append(y)

        dominant_label = {
            neuron: Counter(labels).most_common(1)[0][0]
            for neuron, labels in neuron_labels.items()
        }

        # ── 2. testar ──
        correct = 0

        for x, y in zip(X_test, y_test):
            bmu = self.find_bmu(x)
            pred = dominant_label.get(bmu, -1)

            if pred == y:
                correct += 1

        return correct / len(y_test)

    def train(self, X_train, num_epoch=100, X_test=None, y_train=None, y_test=None):
        """
        Treina a SOM por num_epochs épocas.
        Args:
            data: array de dados de entrada (sem labels).
            num_epochs: número de épocas de treinamento.
            labeled_data: tupla (X, y) com dados e seus verdadeiros labels para
                          calcular a acurácia ao final de cada época. Opcional.
        """

        num_sample = X_train.shape[0]
        max_interactions = num_epoch * num_sample

        # opcional: histórico de acurácia
        self.history_accuracy = []

        for epoch in range(num_epoch):
            # embaralha apenas o treino
            indices = np.arange(num_sample)
            np.random.shuffle(indices)

            iter_start = epoch * num_sample
            epoch_lr = self._decay(iter_start, max_interactions)
            epoch_sigma = self.neighborhood_decay(iter_start, max_interactions)

            for k, i in enumerate(indices):
                interaction = epoch * num_sample + k

                x = X_train[i]
                bmu_idx = self.find_bmu(x)

                curr_lr = self._decay(interaction, max_interactions)
                curr_sigma = self.neighborhood_decay(interaction, max_interactions)

                self._update_weights(x, bmu_idx, curr_lr, curr_sigma)

            # métricas não supervisionadas
            self.history.append(self.weights.copy())
            self.history_lr.append(epoch_lr)
            self.history_sigma.append(epoch_sigma)

            mse = self._compute_mse(X_train)
            self.history_mse.append(mse)

            # 🔹 acurácia correta (sem vazamento)
            if X_test is not None and y_train is not None and y_test is not None:
                acc = self.compute_accuracy(X_train, y_train, X_test, y_test)
                self.history_accuracy.append(acc)

                print(
                    f"Epoca {epoch+1}/{num_epoch} | "
                    f"LR={epoch_lr:.4f} | sigma={epoch_sigma:.4f} | "
                    f"MSE={mse:.4f} | Acc(test)={acc:.2%}"
                )
            else:
                print(
                    f"Epoca {epoch+1}/{num_epoch} | "
                    f"LR={epoch_lr:.4f} | sigma={epoch_sigma:.4f} | "
                    f"MSE={mse:.4f}"
                )

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

        node_label_map = {
            (r, c): [] for r in range(self.line) for c in range(self.column)
        }

        for x, label in zip(data, true_labels):
            bmu_idx = self.find_bmu(x)
            pos = self.grid_positions[bmu_idx]
            node_label_map[(pos[0], pos[1])].append(label)

        label_grid = np.full((self.line, self.column), -1, dtype=int)
        for (row, col), labels in node_label_map.items():
            if labels:
                label_grid[row, col] = Counter(labels).most_common(1)[0][0]

        return label_grid, node_label_map

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
        ax.plot(
            x,
            self.history_mse,
            color="crimson",
            linewidth=2.5,
            marker="o",
            markersize=5,
            label="EQM",
        )
        ax.set_title(
            "Erro Quadrático Médio (EQM) por Época", fontsize=13, fontweight="bold"
        )
        ax.set_xlabel("Época", fontsize=11)
        ax.set_ylabel("EQM médio", fontsize=11)
        # ax.set_xticks(x)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=120, bbox_inches="tight")
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
        avg_acc = sum(acc_pct) / len(acc_pct)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(
            x,
            acc_pct,
            color="steelblue",
            linewidth=2.5,
            marker="s",
            markersize=5,
            label="Acurácia (%)",
        )
        ax.axhline(
            y=avg_acc,
            color="tomato",
            linestyle="--",
            linewidth=2,
            label=f"Acurácia Média ({avg_acc:.2f}%)",
        )
        ax.set_title("Acurácia por Época", fontsize=13, fontweight="bold")
        ax.set_xlabel("Época", fontsize=11)
        ax.set_ylabel("Acurácia (%)", fontsize=11)
        # ax.set_ylim(0, 105)
        # ax.set_xticks(x)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=120, bbox_inches="tight")
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
            fig.suptitle(
                "Decaimento dos Parâmetros da SOM ao Longo das Épocas",
                fontsize=13,
                fontweight="bold",
            )
            x_vals = list(range(1, len(self.history_sigma) + 1))
            sigma_vals = self.history_sigma
            lr_vals = self.history_lr
            x_label = "Época"
            marker = "o"
        else:
            fig.suptitle(
                "Decaimento dos Parâmetros da SOM ao Longo das Iterações",
                fontsize=13,
                fontweight="bold",
            )
            if max_iterations is None:
                print("Aviso: sem histórico e max_iterations não providenciado.")
                return
            x_vals = np.arange(0, max_iterations + 1)
            sigma_vals = self.sigma * np.exp(-x_vals / max_iterations)
            lr_vals = self.learning_rate * np.exp(-x_vals / max_iterations)
            x_label = "Iteração"
            marker = None

        # Sigma
        axes[0].plot(
            x_vals,
            sigma_vals,
            color="darkorange",
            linewidth=2.2,
            marker=marker,
            markersize=5,
        )
        axes[0].set_title("Decaimento do Raio de Vizinhança (σ)", fontweight="bold")
        axes[0].set_xlabel(x_label)
        axes[0].set_ylabel("σ (sigma)")
        if has_history and len(x_vals) <= 20:
            axes[0].set_xticks(x_vals)
        axes[0].grid(True, linestyle="--", alpha=0.4)

        # Learning rate
        axes[1].plot(
            x_vals, lr_vals, color="purple", linewidth=2.2, marker=marker, markersize=2
        )
        axes[1].set_title("Decaimento da Taxa de Aprendizado (α)", fontweight="bold")
        axes[1].set_xlabel(x_label)
        axes[1].set_ylabel("α (learning rate)")
        if has_history and len(x_vals) <= 20:
            axes[1].set_xticks(x_vals)
        axes[1].grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=120, bbox_inches="tight")
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
        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True
        )
        experiment = {
            "seed": int(seed),
            "initial_weights": initial_weights.tolist(),
            "sample_order": [int(i) for i in sample_order],
            "config": config or {},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(experiment, f, indent=2, ensure_ascii=False)
        print(f"Topologia do experimento salva em: {path}")

    @staticmethod
    def load_experiment(path):
        """
        Carrega a topologia de um experimento salvo anteriormente.
        Returns:
            dict com 'seed', 'initial_weights' (np.array), 'sample_order' (list) e 'config' (dict).
        """
        with open(path, "r", encoding="utf-8") as f:
            experiment = json.load(f)
        experiment["initial_weights"] = np.array(experiment["initial_weights"])
        print(f"Topologia carregada de: {path}")
        return experiment
