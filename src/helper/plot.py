import sys
import os
# Adicionando o diretório raiz ao path para que o Python encontre o módulo 'data' e 'src' tranquilamente
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────── Funções de Plotagem ───────────────────────────

SAVE_DIR = "src/pictures"

def plot_progression(history, data, som, indices_to_plot, dim0=0, dim1=5, xlabel="Horas em Redes Sociais (norm.)", ylabel="Nível de Stress (norm.)", save_dir=SAVE_DIR):
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

        # Dados reais projetados nas 2 features escolhidas
        ax.scatter(data[:, dim0], data[:, dim1], c='lightblue', s=15, alpha=0.6, label='Dados' if idx == 0 else "")

        # ==== Posição Anterior dos Neurônios (Azul) ====
        if idx > 0:
            previous_step = indices_to_plot[idx - 1]
            snapshot_prev = history[previous_step]
            for (i_n, j_n) in adjacency:
                ax.plot([snapshot_prev[i_n, dim0], snapshot_prev[j_n, dim0]],
                        [snapshot_prev[i_n, dim1], snapshot_prev[j_n, dim1]],
                        color='blue', linewidth=0.8, alpha=0.3, linestyle='--')
            ax.scatter(snapshot_prev[:, dim0], snapshot_prev[:, dim1],
                       c='blue', s=30, alpha=0.3, marker='s',
                       label='Posição Anterior' if idx == 1 else "")

        # ==== Posição Atual (Vermelho) ====
        for (i_n, j_n) in adjacency:
            ax.plot([snapshot[i_n, dim0], snapshot[j_n, dim0]],
                    [snapshot[i_n, dim1], snapshot[j_n, dim1]],
                    'gray', linewidth=1.0, alpha=0.6)
        ax.scatter(snapshot[:, dim0], snapshot[:, dim1],
                   c='crimson', s=50, zorder=5, edgecolors='black', linewidth=0.7,
                   label='Estado Atual' if idx == 0 else "")

        if step == 0:
            ax.set_title("Estado Inicial", fontsize=11, fontweight='bold')
        else:
            ax.set_title(f"Após Época {step}", fontsize=11, fontweight='bold')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if idx == 0:
            ax.legend(loc="upper left")

    for i in range(len(indices_to_plot), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Evolução da SOM – Teen Mental Health (Progresso por Épocas)", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/mental_progression_epochs.png", dpi=120, bbox_inches='tight')
    print(f"  -> Grafico de progressao salvo: {SAVE_DIR}/mental_progression_epochs.png")
    plt.show()


def plot_cluster_grid(som, cluster_labels, true_labels, data, class_names, dim0=0, dim1=5, save_dir=SAVE_DIR):
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
    plt.savefig(f"{SAVE_DIR}/mental_cluster_grid.png", dpi=120, bbox_inches='tight')
    print(f"  -> Mapa de grade salvo: {SAVE_DIR}/mental_cluster_grid.png")
    plt.show()


def plot_final_clusters(som, data, kmeans, labels, true_labels, class_names, dim0=0, dim1=5, xlabel="Horas/dia em redes sociais (norm.)", ylabel="Nível de stress (norm.)", title_prefix="Estado Final dos Neurônios", save_dir=SAVE_DIR):
    """
    Plota o estado FINAL dos neurônios e usa K-Means para colorir os grupos,
    projetando nas dimensões dim0 e dim1.
    A legenda usa os nomes (class_names) da classe dominante em cada cluster.
    """
    import matplotlib.pyplot as plt
    from collections import Counter
    
    n_clusters = len(set(labels))
    snapshot_final = som.weights.copy()
    
    cores = plt.get_cmap('Set1', n_clusters)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.suptitle(
        f"{title_prefix} – {n_clusters} Clusters (K-Means)\n"
        f"(Eixos: {xlabel}  ↔  {ylabel})",
        fontsize=12, fontweight='bold'
    )
    
    # Dados de fundo
    ax.scatter(data[:, dim0], data[:, dim1],
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
            [snapshot_final[i_n, dim0], snapshot_final[j_n, dim0]],
            [snapshot_final[i_n, dim1], snapshot_final[j_n, dim1]],
            color='gray', linewidth=0.6, alpha=0.4
        )
    
    # Mapear cluster K-Means para o nome da classe dominante
    cluster_to_class = {}
    
    # Descobrir o cluster K-Means de cada amostra (o cluster do BMU dela)
    data_cluster_assignments = []
    for x in data:
        bmu_idx = som.find_bmu(x)
        k_cluster = labels[bmu_idx]
        data_cluster_assignments.append(k_cluster)
        
    for c in range(n_clusters):
        true_labels_in_c = [true_labels[i] for i, k in enumerate(data_cluster_assignments) if k == c]
        if true_labels_in_c:
            dominant_class_idx = Counter(true_labels_in_c).most_common(1)[0][0]
            cluster_to_class[c] = class_names[dominant_class_idx]
        else:
            cluster_to_class[c] = f"Vazio"
            
    # Neurônios coloridos por cluster
    for c in range(n_clusters):
        mask = (np.array(labels) == c)
        cluster_name = cluster_to_class.get(c, "Vazio")
        if np.any(mask):
            ax.scatter(
                snapshot_final[mask, dim0], snapshot_final[mask, dim1],
                color=cores(c), s=80, zorder=5,
                edgecolors='black', linewidths=0.5,
                label=f'{cluster_name} (Cluster {c + 1})'
            )
    
    # Centroides do KMeans (usando features dim0 e dim1)
    # Extract only dim0 and dim1 from centroids if they exist
    centroids = np.array(kmeans.centroids)
    if centroids.shape[0] > 0:
        ax.scatter(
            centroids[:, dim0],
            centroids[:, dim1],
            marker='X', s=200, c='black', zorder=6, label='Centroide'
        )
    

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/clusters_finais.png", dpi=120, bbox_inches='tight')
    print(f"  -> Grafico de clusters em 2D salvo: {SAVE_DIR}/clusters_finais.png")
    plt.show()


def plot_kmeans_comparison(som, data, kmeans_data, kmeans_data_labels, kmeans_neurons, neuron_labels,
                           true_labels, class_names, dim0=0, dim1=1,
                           xlabel="Feature 0", ylabel="Feature 1",
                           save_dir=SAVE_DIR):
    """
    Plota lado a lado:
      - (Esquerda) K-Means aplicado diretamente nos dados brutos
      - (Direita)  K-Means aplicado nos neuronios da SOM (dados coloridos via BMU)
    """
    from collections import Counter

    n_clusters = len(set(kmeans_data_labels))
    cores = plt.get_cmap('Set1', n_clusters)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Comparacao K-Means: Dados Brutos  vs  Neuronios SOM",
                 fontsize=14, fontweight='bold')

    # ────────── Painel Esquerdo: K-Means nos dados brutos ──────────
    ax_l = axes[0]
    ax_l.set_title("K-Means nos Dados Brutos", fontweight='bold')

    # Mapear cluster -> nome da classe dominante
    cluster_to_class_data = {}
    for c in range(n_clusters):
        true_in_c = [true_labels[i] for i, k in enumerate(kmeans_data_labels) if k == c]
        if true_in_c:
            cluster_to_class_data[c] = class_names[Counter(true_in_c).most_common(1)[0][0]]
        else:
            cluster_to_class_data[c] = "Vazio"

    for c in range(n_clusters):
        mask = (np.array(kmeans_data_labels) == c)
        name = cluster_to_class_data.get(c, "Vazio")
        if np.any(mask):
            ax_l.scatter(data[mask, dim0], data[mask, dim1],
                         color=cores(c), s=20, alpha=0.6,
                         label=f'{name} (Cluster {c+1})')

    # Centroides dos dados
    centroids_data = np.array(kmeans_data.centroids)
    if centroids_data.shape[0] > 0:
        ax_l.scatter(centroids_data[:, dim0], centroids_data[:, dim1],
                     marker='X', s=200, c='black', zorder=6, label='Centroide')

    ax_l.set_xlabel(xlabel)
    ax_l.set_ylabel(ylabel)
    ax_l.legend(loc='upper right', fontsize=8)
    ax_l.grid(True, linestyle='--', alpha=0.3)

    # ────────── Painel Direito: K-Means nos neuronios da SOM ──────────
    ax_r = axes[1]
    ax_r.set_title("K-Means nos Neuronios SOM", fontweight='bold')

    snapshot_final = som.weights.copy()

    # Atribuir cada amostra ao cluster do seu BMU
    data_cluster_via_som = []
    for x in data:
        bmu_idx = som.find_bmu(x)
        data_cluster_via_som.append(neuron_labels[bmu_idx])

    # Mapear cluster -> nome da classe dominante (via neuronios)
    cluster_to_class_neurons = {}
    for c in range(n_clusters):
        true_in_c = [true_labels[i] for i, k in enumerate(data_cluster_via_som) if k == c]
        if true_in_c:
            cluster_to_class_neurons[c] = class_names[Counter(true_in_c).most_common(1)[0][0]]
        else:
            cluster_to_class_neurons[c] = "Vazio"

    # Dados de fundo coloridos pelo cluster do BMU
    for c in range(n_clusters):
        mask = (np.array(data_cluster_via_som) == c)
        name = cluster_to_class_neurons.get(c, "Vazio")
        if np.any(mask):
            ax_r.scatter(data[mask, dim0], data[mask, dim1],
                         color=cores(c), s=20, alpha=0.4,
                         label=f'{name} (Cluster {c+1})')

    # Topologia: linhas entre neuronios vizinhos
    adjacency = []
    for i_node, node in enumerate(som.nodes):
        for neighbor in node.neighbors:
            j_node = som.nodes.index(neighbor)
            if i_node < j_node:
                adjacency.append((i_node, j_node))

    for (i_n, j_n) in adjacency:
        ax_r.plot([snapshot_final[i_n, dim0], snapshot_final[j_n, dim0]],
                  [snapshot_final[i_n, dim1], snapshot_final[j_n, dim1]],
                  color='gray', linewidth=1.0, alpha=0.5)

    # Neuronios coloridos por cluster
    for c in range(n_clusters):
        mask = (np.array(neuron_labels) == c)
        name = cluster_to_class_neurons.get(c, "Vazio")
        if np.any(mask):
            ax_r.scatter(snapshot_final[mask, dim0], snapshot_final[mask, dim1],
                         color=cores(c), s=120, zorder=5,
                         edgecolors='black', linewidths=1.0, marker='s',
                         label=f'Neuronios - {name}')

    # Centroides dos neuronios
    centroids_neurons = np.array(kmeans_neurons.centroids)
    if centroids_neurons.shape[0] > 0:
        ax_r.scatter(centroids_neurons[:, dim0], centroids_neurons[:, dim1],
                     marker='X', s=200, c='black', zorder=6, label='Centroide')

    ax_r.set_xlabel(xlabel)
    ax_r.set_ylabel(ylabel)
    ax_r.legend(loc='upper right', fontsize=8)
    ax_r.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/kmeans_comparison.png", dpi=120, bbox_inches='tight')
    print(f"  -> Comparacao K-Means salva: {SAVE_DIR}/kmeans_comparison.png")
    plt.show()
