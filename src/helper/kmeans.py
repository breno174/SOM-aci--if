import numpy as np

class SimpleKMeans:

    def __init__(self, k=3, epochs=50):

        self.k = k
        self.epochs = epochs
        self.centroids = []

    def _distance(self, a, b):
        soma = 0
        for i in range(len(a)):
            soma += (a[i] - b[i]) ** 2
        return math.sqrt(soma)

    def fit(self, data):
        self.centroids = random.sample(data, self.k)
        for epoch in range(self.epochs):
            clusters = [[] for _ in range(self.k)]
            # atribuir pontos
            for point in data:
                min_dist = float("inf")
                idx = 0
                for i, centroid in enumerate(self.centroids):
                    d = self._distance(point, centroid)
                    if d < min_dist:
                        min_dist = d
                        idx = i
                clusters[idx].append(point)
            # atualizar centroides
            new_centroids = []
            for cluster in clusters:
                if len(cluster) == 0:
                    new_centroids.append(random.choice(data))
                    continue
                mean = [0] * len(cluster[0])
                for point in cluster:
                    for j in range(len(point)):
                        mean[j] += point[j]
                for j in range(len(mean)):
                    mean[j] /= len(cluster)
                new_centroids.append(mean)
            self.centroids = new_centroids
        return clusters

class KMeans:
    """
    Implementação do algoritmo K-Means para agrupamento de dados.
    """
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k              # Número de clusters
        self.max_iters = max_iters # Número máximo de iterações
        self.tol = tol          # Tolerância para convergência (mudança nos centroides)
        self.centroids = None   # Coordenadas dos centroides
        self.labels = None      # Índice do cluster para cada ponto de dado

    def fit(self, X):
        """
        Treina o modelo K-Means com os dados X.
        """
        # 1. Inicialização: Escolhe k pontos aleatórios como centroides iniciais
        n_samples, n_features = X.shape
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices]

        for i in range(self.max_iters):
            # 2. Atribuição: Encontra o centroide mais próximo para cada ponto
            old_centroids = self.centroids.copy()
            self.labels = self._assign_clusters(X)

            # 3. Atualização: Recalcula os centroides baseados na média dos pontos atribuídos
            self.centroids = self._update_centroids(X)

            # 4. Verificação de Convergência: Se a distância total de mudança for menor que a tolerância
            center_shift = np.linalg.norm(self.centroids - old_centroids)
            if center_shift < self.tol:
                print(f"K-Means convergiu na iteração {i}.")
                break

        return self.centroids, self.labels

    def fit_predict(self, X):
        """
        Executa o fit e retorna as labels.
        """
        self.fit(X)
        return self.labels

    def _assign_clusters(self, X):
        """
        Calcula a distância de cada ponto para cada centroide e retorna o índice do mais próximo.
        """
        # Calcula distâncias euclidianas (n_samples, k)
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X):
        """
        Calcula a média de todos os pontos de cada cluster para definir o novo centroide.
        """
        new_centroids = np.zeros((self.k, X.shape[1]))
        for i in range(self.k):
            points = X[self.labels == i]
            if len(points) > 0:
                new_centroids[i] = np.mean(points, axis=0)
            else:
                # Se um cluster ficar vazio, mantém o centroide anterior ou escolhe um novo ponto aleatório
                new_centroids[i] = X[np.random.choice(len(X))]
        return new_centroids

    def predict(self, X):
        """
        Atribui novos pontos aos clusters existentes.
        """
        return self._assign_clusters(X)
