import numpy as np


class KMeans:
    def __init__(self, k=3, max_iters=100, tol=1e-4, random_state=None):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state

        self.centroids = None
        self.labels = None
        self.inertia_ = None  # soma das distâncias quadráticas (SSE)

    # ─────────────────────────────
    # Treinamento
    # ─────────────────────────────
    def fit(self, X):
        X = np.array(X, dtype=float)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape

        # 1. Inicialização (amostras aleatórias)
        indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[indices]

        for _ in range(self.max_iters):
            old_centroids = self.centroids.copy()

            # 2. Atribuição
            self.labels = self._assign_clusters(X)

            # 3. Atualização
            self.centroids = self._update_centroids(X)

            # 4. Convergência
            shift = np.linalg.norm(self.centroids - old_centroids)

            if shift < self.tol:
                break

        # 5. Inertia (SSE)
        self.inertia_ = self._compute_inertia(X)

        return self.centroids, self.labels

    # ─────────────────────────────
    # Atribuição vetorizada
    # ─────────────────────────────
    def _assign_clusters(self, X):
        # shape: (n_samples, k)
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    # ─────────────────────────────
    # Atualização vetorizada
    # ─────────────────────────────
    def _update_centroids(self, X):
        new_centroids = np.zeros_like(self.centroids)

        for i in range(self.k):
            points = X[self.labels == i]

            if len(points) > 0:
                new_centroids[i] = np.mean(points, axis=0)
            else:
                # evita cluster vazio
                new_centroids[i] = X[np.random.choice(len(X))]

        return new_centroids

    # ─────────────────────────────
    # Inertia (SSE)
    # ─────────────────────────────
    def _compute_inertia(self, X):
        distances = np.linalg.norm(X - self.centroids[self.labels], axis=1)
        return np.sum(distances ** 2)

    # ─────────────────────────────
    # Predict
    # ─────────────────────────────
    def predict(self, X):
        X = np.array(X, dtype=float)
        return self._assign_clusters(X)

    # ─────────────────────────────
    # Fit + Predict
    # ─────────────────────────────
    def fit_predict(self, X):
        self.fit(X)
        return self.labels