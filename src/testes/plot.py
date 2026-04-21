import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from src.helper.kmeans import KMeans

from src.som import SOM
from data.normalize import Normalize


def preparar_dados():
    """Carrega e normaliza apenas 2 colunas do dataset para plotagem 2D."""
    norm = Normalize("data/Teen_Mental_Health_Dataset.csv")
    # Carregar todas as colunas que precisamos (as 2 que vamos usar)
    norm.load_with_pandas(["daily_social_media_hours", "stress_level"])
    data = np.array(norm.data, dtype=float)
    
    # Normalização Min-Max para [0, 1]
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    diff = data_max - data_min
    diff[diff == 0] = 1
    data = (data - data_min) / diff
    
    return data, data_min, data_max


def plotar_epocas(som, data, num_epochs):
    """
    Plota a posição dos neurônios a cada época em subgráficos separados.
    Cada neurônio é mostrado como um ponto, com linhas conectando os vizinhos.
    """
    cols = min(num_epochs, 5)
    rows = (num_epochs + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.array(axes).flatten()
    
    fig.suptitle(
        "Evolução dos Neurônios da SOM por Época\n"
        "(Eixos: Horas em Redes Sociais  ↔  Nível de Stress)",
        fontsize=13, fontweight='bold'
    )
    
    for epoch_idx, snapshot in enumerate(som.history):
        ax = axes[epoch_idx]
        
        # Dados de entrada (fundo)
        ax.scatter(data[:, 0], data[:, 1], c='lightblue', s=10, alpha=0.4,
                   label='Dados' if epoch_idx == 0 else None)
        
        # Conexões entre neurônios vizinhos (mostra topologia da grade)
        for node in som.nodes:
            w_node = snapshot[som.nodes.index(node)]
            for neighbor in node.neighbors:
                w_neighbor = snapshot[som.nodes.index(neighbor)]
                ax.plot(
                    [w_node[0], w_neighbor[0]],
                    [w_node[1], w_neighbor[1]],
                    'gray', linewidth=0.5, alpha=0.5
                )
        
        # Posição dos neurônios
        ax.scatter(snapshot[:, 0], snapshot[:, 1],
                   c='crimson', s=40, zorder=5, label='Neurônios')
        
        ax.set_title(f"Época {epoch_idx + 1}", fontsize=10)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Horas/dia em redes sociais (norm.)")
        ax.set_ylabel("Nível de stress (norm.)")
        ax.grid(True, linestyle='--', alpha=0.3)
    
    # Ocultar subgráficos extras
    for i in range(len(som.history), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig("src/evolucao_epocas.png", dpi=120, bbox_inches='tight')
    print("  → Gráfico de épocas salvo: src/evolucao_epocas.png")
    plt.show()


def plotar_clusters(som, data, n_clusters=3):
    """
    Plota o estado FINAL dos neurônios e usa KMeans para sugerir clusters,
    mostrando visualmente se os neurônios se organizaram em grupos distintos.
    """
    snapshot_final = som.history[-1]
    
    # KMeans sobre as posições finais dos neurônios
    kmeans = KMeans(k=n_clusters)
    labels = kmeans.fit_predict(snapshot_final)
    
    cores = cm.get_cmap('Set1', n_clusters)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.suptitle(
        f"Estado Final dos Neurônios – {n_clusters} Clusters (KMeans)\n"
        "(Eixos: Horas em Redes Sociais  ↔  Nível de Stress)",
        fontsize=12, fontweight='bold'
    )
    
    # Dados de fundo
    ax.scatter(data[:, 0], data[:, 1],
               c='lightgray', s=12, alpha=0.5, label='Dados reais')
    
    # Conexões entre vizinhos coloridas pelo cluster do neurônio de origem
    for node in som.nodes:
        idx = som.nodes.index(node)
        w_node = snapshot_final[idx]
        cluster_cor = cores(labels[idx])
        for neighbor in node.neighbors:
            n_idx = som.nodes.index(neighbor)
            w_neighbor = snapshot_final[n_idx]
            ax.plot(
                [w_node[0], w_neighbor[0]],
                [w_node[1], w_neighbor[1]],
                color='gray', linewidth=0.6, alpha=0.4
            )
    
    # Neurônios coloridos por cluster
    for c in range(n_clusters):
        mask = labels == c
        ax.scatter(
            snapshot_final[mask, 0], snapshot_final[mask, 1],
            color=cores(c), s=80, zorder=5,
            edgecolors='black', linewidths=0.5,
            label=f'Cluster {c + 1}'
        )
    
    # Centroides do KMeans
    ax.scatter(
        kmeans.centroids[:, 0],
        kmeans.centroids[:, 1],
        marker='X', s=200, c='black', zorder=6, label='Centroide'
    )
    
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Horas/dia em redes sociais (norm.)")
    ax.set_ylabel("Nível de stress (norm.)")
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("src/clusters_finais.png", dpi=120, bbox_inches='tight')
    print("  → Gráfico de clusters salvo: src/clusters_finais.png")
    plt.show()


def plotar_taxa_aprendizado(som):
    """
    Plota a curva de decaimento da taxa de aprendizado por época.
    Mostra como o modelo vai "desacelerando" ao longo do treino.
    """
    epocas = list(range(1, len(som.history_lr) + 1))
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epocas, som.history_lr,
            color='steelblue', linewidth=2.5, marker='o', markersize=6,
            label='Taxa de Aprendizado')
    ax.fill_between(epocas, som.history_lr, alpha=0.15, color='steelblue')
    
    ax.set_title("Decaimento da Taxa de Aprendizado por Época", fontsize=13, fontweight='bold')
    ax.set_xlabel("Época", fontsize=11)
    ax.set_ylabel("Taxa de Aprendizado (α)", fontsize=11)
    ax.set_xticks(epocas)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig("src/taxa_aprendizado.png", dpi=120, bbox_inches='tight')
    print("  → Gráfico de taxa de aprendizado salvo: src/taxa_aprendizado.png")
    plt.show()


def plotar_eqm(som):
    """
    Plota o Erro Quadrático Médio (EQM / Quantization Error) por época.
    Um EQM decrescente indica que os neurônios estão representando
    cada vez melhor a distribuição dos dados.
    """
    epocas = list(range(1, len(som.history_mse) + 1))
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epocas, som.history_mse,
            color='crimson', linewidth=2.5, marker='s', markersize=6,
            label='EQM (Quantization Error)')
    ax.fill_between(epocas, som.history_mse, alpha=0.15, color='crimson')
    
    ax.set_title("Erro Quadrático Médio (EQM) por Época", fontsize=13, fontweight='bold')
    ax.set_xlabel("Época", fontsize=11)
    ax.set_ylabel("EQM médio por amostra", fontsize=11)
    ax.set_xticks(epocas)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig("src/eqm.png", dpi=120, bbox_inches='tight')
    print("  → Gráfico de EQM salvo: src/eqm.png")
    plt.show()


def main():
    NUM_EPOCHS = 10
    GRID_M, GRID_N = 5, 5  # grade menor facilita enxergar os neurônios no gráfico 2D
    N_CLUSTERS = 3
    
    print("[1/5] Carregando e normalizando dados (2 colunas)...")
    data, _, _ = preparar_dados()
    print(f"      {data.shape[0]} amostras | dim=2")
    
    print("[2/5] Treinando SOM...")
    origin = np.zeros(2)  # Ponto de origem no (0, 0)
    som = SOM(line=GRID_M, column=GRID_N, dim=2,
              learning_rate=0.1, initial_weights=origin)
    som.train(data, num_epochs=NUM_EPOCHS)
    print("      Treinamento concluído!")
    
    print("[3/5] Plotando evolução das épocas...")
    plotar_epocas(som, data, NUM_EPOCHS)
    
    print("[4/5] Plotando taxa de aprendizado e EQM...")
    plotar_taxa_aprendizado(som)
    plotar_eqm(som)
    
    print("[5/5] Plotando clusters finais...")
    plotar_clusters(som, data, n_clusters=N_CLUSTERS)
    
    print("Pronto!")


if __name__ == "__main__":
    main()
