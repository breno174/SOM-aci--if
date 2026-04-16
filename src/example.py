import sys
import os
# Adicionando o diretório raiz ao path para que o Python encontre o módulo 'data' e 'src' tranquilamente
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd

from src.som import SOM
from src.helper.kmeans import KMeans
from src.plot import plotar_epocas, plotar_taxa_aprendizado, plotar_eqm, plotar_clusters
from data.normalize import Normalize


# ─────────────────────────── Funções de Plotagem ───────────────────────────

def plot_progression(history, data, som, indices_to_plot):
    """Plota a progressão dos neurônios época a época."""
    num_plots = len(indices_to_plot)
    cols = 4
    rows = (num_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.array(axes).flatten()

    # Pré-computar adjacency para desenhar linhas de topologia
    n_neurons = som.num_neurons
    adjacency = []
    for i_node, node in enumerate(som.nodes):
        for neighbor in node.neighbors:
            j_node = som.nodes.index(neighbor)
            if i_node < j_node:  # evita duplicar linhas
                adjacency.append((i_node, j_node))

    for idx, (ax, step) in enumerate(zip(axes, indices_to_plot)):
        snapshot = history[step]

        # Dados reais projetados nas primeiras 2 features (social media hours x stress_level)
        ax.scatter(data[:, 0], data[:, 5], c='lightblue', s=15, alpha=0.6, label='Dados' if idx == 0 else "")

        # ==== Posição Anterior dos Neurônios (Azul) ====
        if idx > 0:
            previous_step = indices_to_plot[idx - 1]
            snapshot_prev = history[previous_step]
            for (i_n, j_n) in adjacency:
                ax.plot([snapshot_prev[i_n, 0], snapshot_prev[j_n, 0]],
                        [snapshot_prev[i_n, 5], snapshot_prev[j_n, 5]],
                        color='blue', linewidth=0.8, alpha=0.3, linestyle='--')
            ax.scatter(snapshot_prev[:, 0], snapshot_prev[:, 5],
                       c='blue', s=30, alpha=0.3, marker='s',
                       label='Posição Anterior' if idx == 1 else "")

        # ==== Posição Atual (Vermelho) ====
        for (i_n, j_n) in adjacency:
            ax.plot([snapshot[i_n, 0], snapshot[j_n, 0]],
                    [snapshot[i_n, 5], snapshot[j_n, 5]],
                    'gray', linewidth=1.0, alpha=0.6)
        ax.scatter(snapshot[:, 0], snapshot[:, 5],
                   c='crimson', s=50, zorder=5, edgecolors='black', linewidth=0.7,
                   label='Estado Atual' if idx == 0 else "")

        if step == 0:
            ax.set_title("Estado Inicial", fontsize=11, fontweight='bold')
        else:
            ax.set_title(f"Após Época {step}", fontsize=11, fontweight='bold')
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel("Horas em Redes Sociais (norm.)")
        ax.set_ylabel("Nível de Stress (norm.)")
        if idx == 0:
            ax.legend(loc="upper left")

    for i in range(len(indices_to_plot), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Evolução da SOM – Teen Mental Health (Progresso por Épocas)", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("src/mental_progression_epochs.png", dpi=120, bbox_inches='tight')
    print("  → Gráfico de progressão salvo: src/mental_progression_epochs.png")
    plt.show()


def plot_cluster_grid(som, cluster_labels, true_labels, data, class_names, dim0=0, dim1=5):
    """Plota o mapa de grade da SOM com clusters K-Means e labels reais."""
    cluster_grid = som.get_cluster_grid(cluster_labels)
    label_grid, node_counts = som.label_nodes_by_data(data, true_labels)

    num_clusters = len(set(cluster_labels))
    num_classes = len(class_names)
    cluster_colors = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12', '#9b59b6']
    class_colors   = ['#1abc9c', '#e67e22', '#2980b9', '#8e44ad', '#c0392b', '#27ae60']

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Mapa de Grade da SOM – Teen Mental Health", fontsize=14, fontweight='bold')

    from matplotlib.patches import Patch
    for ax, grid, colors, title, names in [
        (axes[0], cluster_grid, cluster_colors, "Clusters K-Means por Neurônio",
         [f'Cluster {i+1}' for i in range(num_clusters)]),
        (axes[1], label_grid, class_colors, "Label Dominante por Neurônio (BMU)", class_names),
    ]:
        ax.set_title(title, fontweight='bold')
        ax.set_xlim(-0.5, som.column - 0.5)
        ax.set_ylim(-0.5, som.line - 0.5)
        ax.set_xticks(range(som.column))
        ax.set_yticks(range(som.line))
        ax.invert_yaxis()
        ax.grid(True, linestyle='--', alpha=0.3)

        for row in range(som.line):
            for col in range(som.column):
                cid = grid[row, col]
                color = colors[cid % len(colors)] if cid >= 0 else '#cccccc'
                ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, color=color, alpha=0.7))
                label_txt = f"C{cid+1}" if grid is cluster_grid else (names[cid] if cid >= 0 else '?')
                count = len(node_counts.get((row, col), [])) if grid is label_grid else ''
                txt = f"{label_txt}\n({count})" if grid is label_grid else label_txt
                ax.text(col, row, txt, ha='center', va='center', fontsize=7, color='white', fontweight='bold')

        legend_handles = [Patch(facecolor=colors[i % len(colors)], label=names[i]) for i in range(len(names))]
        ax.legend(handles=legend_handles, loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig("src/mental_cluster_grid.png", dpi=120, bbox_inches='tight')
    print("  → Mapa de grade salvo: src/mental_cluster_grid.png")
    plt.show()


def plot_final_clusters(som, data, kmeans, labels):
    """
    Plota o estado FINAL dos neurônios e usa K-Means para colorir os grupos,
    projetando nas dimensões 0 (sociais) e 5 (stress) da mesma forma que plot_progression.
    """
    import matplotlib.cm as cm
    n_clusters = len(set(labels))
    snapshot_final = som.weights.copy()
    
    cores = cm.get_cmap('Set1', n_clusters)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.suptitle(
        f"Estado Final dos Neurônios – {n_clusters} Clusters (K-Means)\n"
        "(Eixos: Horas em Redes Sociais  ↔  Nível de Stress)",
        fontsize=12, fontweight='bold'
    )
    
    # Dados de fundo
    ax.scatter(data[:, 0], data[:, 5],
               c='lightgray', s=12, alpha=0.5, label='Dados reais')
    
    # Pré-computar adjacency
    adjacency = []
    for i_node, node in enumerate(som.nodes):
        for neighbor in node.neighbors:
            j_node = som.nodes.index(neighbor)
            if i_node < j_node:
                adjacency.append((i_node, j_node))
                
    # Conexões entre vizinhos
    for (i_n, j_n) in adjacency:
        ax.plot(
            [snapshot_final[i_n, 0], snapshot_final[j_n, 0]],
            [snapshot_final[i_n, 5], snapshot_final[j_n, 5]],
            color='gray', linewidth=0.6, alpha=0.4
        )
    
    # Neurônios coloridos por cluster
    for c in range(n_clusters):
        mask = (np.array(labels) == c)
        ax.scatter(
            snapshot_final[mask, 0], snapshot_final[mask, 5],
            color=cores(c), s=80, zorder=5,
            edgecolors='black', linewidths=0.5,
            label=f'Cluster {c + 1}'
        )
    
    # Centroides do KMeans (usando features 0 e 5)
    ax.scatter(
        kmeans.centroids[:, 0],
        kmeans.centroids[:, 5],
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
    print("  → Gráfico de clusters em 2D salvo: src/clusters_finais.png")
    plt.show()

# ─────────────────────────────── Main ───────────────────────────────────────

def main():
    # 1. Carregar e normalizar dados
    print("Carregando o dataset Teen Mental Health...")
    norm = Normalize("data/Teen_Mental_Health_Dataset.csv")
    data_raw = norm.usage_coluns()  # retorna numpy com colunas numéricas + platform_usage como int
    data = np.array(data_raw, dtype=float)
    data = np.nan_to_num(data)

    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    diff = data_max - data_min
    diff[diff == 0] = 1
    data = (data - data_min) / diff
    print(f"  {data.shape[0]} amostras | dim={data.shape[1]}")

    # Também carregar stress_level como label ordinal (0=Low, 1=Medium, 2=High)
    df_full = pd.read_csv("data/Teen_Mental_Health_Dataset.csv")
    stress_map = {v: i for i, v in enumerate(sorted(df_full['stress_level'].unique()))}
    stress_int = df_full['stress_level'].map(stress_map).values
    stress_names = [k for k, _ in sorted(stress_map.items(), key=lambda x: x[1])]
    num_classes = len(stress_names)
    print(f"  Classes de stress: {stress_names}")

    # 2. Configurar e inicializar SOM
    dim = data.shape[1]
    m, n = 5, 5
    print(f"\nInicializando SOM {m}x{n} (Dimensão do Neurônio: {dim}).")
    origin = np.zeros(dim)
    som = SOM(line=m, column=n, dim=dim, learning_rate=0.3, sigma=2,
              initial_weights=origin)

    # 3. Treinar SOM por todas as épocas via som.train()
    print("\nExecutando treinamento e registrando progressão por época...")
    num_epochs = 100
    num_samples = data.shape[0]
    max_iterations = num_epochs * num_samples

    seed = int(np.random.randint(0, 2**31 - 1))
    np.random.seed(seed)
    print(f"  Semente aleatória: {seed}")

    initial_weights_snapshot = som.weights.copy()
    
    som.train(data, num_epochs=num_epochs, labeled_data=(data, stress_int))
    print("  Treinamento completo concluído!")

    # Juntar o estado inicial com os estados salvos a cada época
    history_per_epoch = [initial_weights_snapshot.copy()] + som.history

    # 4. Salvar topologia do experimento
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_path = f"src/experiments/mental_{timestamp}.json"
    SOM.save_experiment(
        path=experiment_path,
        seed=seed,
        initial_weights=initial_weights_snapshot,
        sample_order=[], # Não mantemos a ordem de amostras de todas as épocas para simplificar
        config={'grid': [m, n], 'dim': dim, 'learning_rate': 0.3,
                'sigma': max(m, n) / 2.0, 'num_epochs': num_epochs, 'num_samples': num_samples}
    )

    # 5. Plotar progressão por época (entre 6 e 10 gráficos)
    total_snapshots = len(history_per_epoch)
    
    # Encontrar o step adequado para termos de 6 a 10 gráficos
    step = 1
    for s in range(1, total_snapshots):
        num_plots = len(range(0, total_snapshots, s))
        if range(0, total_snapshots, s)[-1] != total_snapshots - 1:
            num_plots += 1
        if 6 <= num_plots <= 10:
            step = s
            break

    indices_to_plot = list(range(0, total_snapshots, step))
    if indices_to_plot[-1] != total_snapshots - 1:
         indices_to_plot.append(total_snapshots - 1)
         
    # Se ainda assim tiver menos de 6 ou mais de 10 por alguma anomalia, força no mínimo 6
    if len(indices_to_plot) < 6:
        indices_to_plot = np.linspace(0, total_snapshots - 1, min(6, total_snapshots), dtype=int).tolist()
    elif len(indices_to_plot) > 10:
        indices_to_plot = np.linspace(0, total_snapshots - 1, 10, dtype=int).tolist()

    # Garantir valores únicos e ordenados
    indices_to_plot = sorted(list(set(indices_to_plot)))

    print(f"\nPlotando progressão por épocas ({len(indices_to_plot)} momentos)...")
    plot_progression(history_per_epoch, data, som, indices_to_plot)

    # 6. EQM, Acurácia e Decaimento (histórico das 20 épocas)
    final_mse = som.history_mse[-1]
    final_acc  = som.history_accuracy[-1]
    print(f"\nErro Quadrático Médio Final (Época {num_epochs}): {final_mse:.4f}")
    print(f"Acurácia Final da SOM (Época {num_epochs}): {final_acc * 100:.1f}%")

    som.plot_mse_history(save_path="src/mental_eqm.png")
    som.plot_accuracy_history(save_path="src/mental_accuracy.png")
    som.plot_neighborhood_decay(max_iterations=max_iterations,
                                save_path="src/mental_neighborhood_decay.png")

    # 6. K-Means sobre os pesos finais
    print("\nExecutando K-Means sobre os pesos finais...")
    final_weights = som.weights.copy()
    kmeans = KMeans(k=num_classes)
    labels = kmeans.fit_predict(final_weights)

    # 7. Mapa de grade com clusters e labels reais
    print("Plotando mapa de grade...")
    plot_cluster_grid(som, labels, stress_int, data, stress_names)

    # 7b. Gráfico de Clusters na projeção 2D (Requisitado)
    print("Plotando K-Means na projeção de Espaço de Features...")
    plot_final_clusters(som, data, kmeans, labels)

    # 8. Matriz de confusão
    print("Gerando matriz de confusão...")
    som.plot_confusion_matrix(
        data=data,
        true_labels=stress_int,
        class_names=stress_names,
        save_path="src/mental_confusion_matrix.png"
    )


if __name__ == "__main__":
    main()
