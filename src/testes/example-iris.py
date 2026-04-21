import sys
import os
# Adicionando o diretório raiz ao path para que o Python encontre o módulo 'data' e 'src' tranquilamente
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.som import SOM
from src.helper.kmeans import KMeans
from data.normalize import Normalize

def plot_progression(history, data, som_nodes, indices_to_plot):
    num_plots = len(indices_to_plot)
    cols = 4
    rows = (num_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.array(axes).flatten()
    for idx, (ax, step) in enumerate(zip(axes, indices_to_plot)):
        snapshot = history[step]
        
        # Usaremos as colunas 2 e 3 (Petal Length/Width) para visualizar os dados em 2D
        # Dados de entrada
        ax.scatter(data[:, 2], data[:, 3], c='lightblue', s=15, alpha=0.6, label='Dados' if idx==0 else "")

        # ==== 1. Posição Anterior dos Neurônios (Fundo Azul) ====
        if idx > 0:
            previous_step = indices_to_plot[idx - 1]
            snapshot_previous = history[previous_step]
            for node in som_nodes:
                idx_node = som_nodes.index(node)
                w_node_i = snapshot_previous[idx_node]
                for neighbor in node.neighbors:
                    n_idx = som_nodes.index(neighbor)
                    w_neighbor_i = snapshot_previous[n_idx]
                    ax.plot([w_node_i[2], w_neighbor_i[2]], [w_node_i[3], w_neighbor_i[3]], 
                            color='blue', linewidth=0.8, alpha=0.3, linestyle='--')
            
            ax.scatter(snapshot_previous[:, 2], snapshot_previous[:, 3], c='blue', s=30, alpha=0.3, marker='s', label=f'Posição Anterior' if idx==1 else "")

        # ==== 2. Posição Atual (Etapa Atual em Vermelho) ====
        for node in som_nodes:
            idx_node = som_nodes.index(node)
            w_node = snapshot[idx_node]
            for neighbor in node.neighbors:
                idx_neighbor = som_nodes.index(neighbor)
                w_neighbor = snapshot[idx_neighbor]
                ax.plot([w_node[2], w_neighbor[2]], [w_node[3], w_neighbor[3]], 'gray', linewidth=1.0, alpha=0.6)
                
        ax.scatter(snapshot[:, 2], snapshot[:, 3], c='crimson', s=50, zorder=5, edgecolors='black', linewidth=0.7, label='Estado Atual' if idx==0 else "")
        
        ax.set_title(f"Após Amostra {step}", fontsize=11, fontweight='bold')
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        
        if idx == 0:
            ax.legend(loc="upper left")
        
    for i in range(len(indices_to_plot), len(axes)):
        axes[i].set_visible(False)
        
    fig.suptitle("Evolução da SOM a cada apresentação de amostra (1 Época)", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("src/iris_progression_1epoch.png", dpi=120, bbox_inches='tight')
    print("Gráfico da progressão salvo em: src/iris_progression_1epoch.png")
    plt.show()

def plot_final_kmeans(weights, labels, data, som_nodes, mse):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Dados reais
    ax.scatter(data[:, 2], data[:, 3], c='lightgray', s=15, alpha=0.5, label='Dados reais')
    
    # Conexões da rede
    for node in som_nodes:
        idx = som_nodes.index(node)
        w_node = weights[idx]
        for neighbor in node.neighbors:
            n_idx = som_nodes.index(neighbor)
            w_neighbor = weights[n_idx]
            ax.plot([w_node[2], w_neighbor[2]], [w_node[3], w_neighbor[3]], 'gray', linewidth=0.5, alpha=0.4)
            
    # Neurônios por cluster
    cores = ['red', 'green', 'blue']
    for c in range(3):
        mask = labels == c
        if np.any(mask):
            ax.scatter(weights[mask, 2], weights[mask, 3], c=cores[c], s=100, edgecolors='black', zorder=5, label=f'Cluster {c+1}')
            
    ax.set_title(f"Estado Final + K-Means (EQM: {mse:.4f})", fontweight='bold')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig("src/iris_kmeans_final.png", dpi=120, bbox_inches='tight')
    print("Gráfico do KMeans salvo em: src/iris_kmeans_final.png")
    plt.show()

def plot_cluster_grid(som, cluster_labels, species_labels, data):
    """
    Plota dois mapas de grade lado a lado:
    - Esquerdo: cada célula colorida pelo cluster K-Means do neurônio.
    - Direito: cada célula colorida pela espécie real dominante das amostras que o BMU capturou.
    """
    cluster_grid = som.get_cluster_grid(cluster_labels)
    label_grid, node_counts = som.label_nodes_by_data(data, species_labels)

    cluster_colors = ['#e74c3c', '#2ecc71', '#3498db']  # Vermelho, Verde, Azul
    species_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    species_colors = ['#f39c12', '#8e44ad', '#1abc9c']  # Laranja, Roxo, Verde-agua

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Mapa de Grade da SOM – Análise de Clusters", fontsize=15, fontweight='bold')

    # ─── Mapa esquerdo: clusters K-Means ───
    ax1 = axes[0]
    ax1.set_title("Clusters K-Means por Neurônio", fontweight='bold')
    ax1.set_xlim(-0.5, som.column - 0.5)
    ax1.set_ylim(-0.5, som.line - 0.5)
    ax1.set_xticks(range(som.column))
    ax1.set_yticks(range(som.line))
    ax1.invert_yaxis()
    ax1.set_xlabel("Coluna")
    ax1.set_ylabel("Linha")
    ax1.grid(True, linestyle='--', alpha=0.3)

    for row in range(som.line):
        for col in range(som.column):
            cid = cluster_grid[row, col]
            color = cluster_colors[cid] if cid >= 0 else '#cccccc'
            rect = plt.Rectangle((col - 0.5, row - 0.5), 1, 1, color=color, alpha=0.7)
            ax1.add_patch(rect)
            ax1.text(col, row, f"C{cid+1}", ha='center', va='center', fontsize=8, fontweight='bold', color='white')

    from matplotlib.patches import Patch
    legend_km = [Patch(facecolor=cluster_colors[i], label=f'Cluster {i+1}') for i in range(3)]
    ax1.legend(handles=legend_km, loc='upper right', fontsize=9)

    # ─── Mapa direito: espécie dominante (ground-truth) ───
    ax2 = axes[1]
    ax2.set_title("Espécie Dominante por Neurônio (BMU)", fontweight='bold')
    ax2.set_xlim(-0.5, som.column - 0.5)
    ax2.set_ylim(-0.5, som.line - 0.5)
    ax2.set_xticks(range(som.column))
    ax2.set_yticks(range(som.line))
    ax2.invert_yaxis()
    ax2.set_xlabel("Coluna")
    ax2.set_ylabel("Linha")
    ax2.grid(True, linestyle='--', alpha=0.3)

    for row in range(som.line):
        for col in range(som.column):
            sid = label_grid[row, col]
            color = species_colors[sid] if sid >= 0 else '#cccccc'
            rect = plt.Rectangle((col - 0.5, row - 0.5), 1, 1, color=color, alpha=0.7)
            ax2.add_patch(rect)
            count = len(node_counts.get((row, col), []))
            short = ['Set', 'Ver', 'Vir', '?'][sid] if sid >= 0 else '?'
            ax2.text(col, row, f"{short}\n({count})", ha='center', va='center', fontsize=7, fontweight='bold', color='white')

    legend_sp = [Patch(facecolor=species_colors[i], label=species_names[i]) for i in range(3)]
    ax2.legend(handles=legend_sp, loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig("src/iris_cluster_grid.png", dpi=120, bbox_inches='tight')
    print("Gráfico de grade de clusters salvo em: src/iris_cluster_grid.png")
    plt.show()

def main():
    # 1. Carregar os dados da Iris
    print("Carregando o dataset Iris...")
    norm = Normalize("data/Iris.csv")
    
    # Selecionamos as 4 colunas numéricas
    columns = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    data = norm.load_with_pandas(columns)
    print(f'data: {data} \n')
    # Forçar conversão para float
    data = np.array(data, dtype=float)
    print(f'data convert float: {data} \n')
    
    # Normalização min-max (colocar features entre 0 e 1)
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    diff = data_max - data_min
    diff[diff == 0] = 1 
    data = (data - data_min) / diff

    # 2. Configurações da SOM para treinamento de 1 época
    dim = data.shape[1]
    print(f'dim: {dim} \n')
    print(f'shape: {data.shape[0]} \n')
    m, n = 3, 2 # Tamanho da Grade
    print(f"Inicializando SOM {m}x{n} (Dimensão do Neurônio: {dim}).")
    
    origin = np.zeros(dim)
    som = SOM(line=m, column=n, dim=dim, learning_rate=0.1, sigma=2, initial_weights=origin)

    # 3. Treinamento Manual Capturando a Evolução a cada Amostra
    print("Iniciando treinamento detalhado por 1 época apenas...")
    num_epochs = 1
    num_samples = data.shape[0]
    max_iterations = num_epochs * num_samples

    # Gerar semente aleatória e aplicar para garantir reprodutibilidade
    seed = int(np.random.randint(0, 2**31 - 1))
    np.random.seed(seed)
    print(f"Semente aleatória usada: {seed}")

    # Salvar os pesos iniciais (antes de qualquer apresentação)
    initial_weights_snapshot = np.array([node.neuron.weights.copy() for node in som.nodes])

    # Embaralhar com índices rastreados para salvar a ordem de apresentação
    shuffled_indices = np.random.permutation(num_samples)
    data_shuffled = data[shuffled_indices]

    history_per_sample = []
    # Salvar o estado inicial (antes de qualquer amostra)
    history_per_sample.append(initial_weights_snapshot.copy())
    
    for i, x in enumerate(data_shuffled):
        bmu = som.find_bmu(x)
        
        # Calcular taxas de decaimento
        curr_lr = som._decay(som.learning_rate, i, max_iterations)
        curr_sigma = som._decay(som.sigma, i, max_iterations)
        
        # Atualizar os pesos
        som._update_weights(x, bmu, curr_lr, curr_sigma)
        
        # Salvar o estado após apresentação da amostra 'i'
        history_per_sample.append(np.array([node.neuron.weights.copy() for node in som.nodes]))

    print("Treinamento de 1 época concluído!")

    # Salvar topologia do experimento para replicação futura
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_path = f"src/experiments/experiment_{timestamp}.json"
    SOM.save_experiment(
        path=experiment_path,
        seed=seed,
        initial_weights=initial_weights_snapshot,
        sample_order=shuffled_indices.tolist(),
        config={
            'grid': [m, n],
            'dim': dim,
            'learning_rate': 0.1,
            'sigma': 2,
            'num_epochs': 1,
            'num_samples': num_samples,
        }
    )

    # Selecionar cerca de 16 momentos para plotar (inclusive o início e o fim)
    if num_samples > 15:
        step = num_samples // 15
        indices_to_plot = list(range(0, num_samples, step))
        if indices_to_plot[-1] != num_samples:
            indices_to_plot.append(num_samples)
    else:
        indices_to_plot = list(range(num_samples + 1))
        
    print(f"Plotando os momentos (interação amostra a amostra): {indices_to_plot}")
    plot_progression(history_per_sample, data, som.nodes, indices_to_plot)

    # 4. Carregar labels reais das espécies (para acurácia e matriz de confusão)
    import pandas as pd
    df_full = pd.read_csv("data/Iris.csv")
    species_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    species_int = df_full['Species'].map(species_map).values
    
    # Normalizar os dados originais (mesmos dados do treino, sem shuffle)
    raw_data = np.array(df_full[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']], dtype=float)
    raw_min = raw_data.min(axis=0)
    raw_max = raw_data.max(axis=0)
    raw_diff = raw_max - raw_min
    raw_diff[raw_diff == 0] = 1
    raw_data_norm = (raw_data - raw_min) / raw_diff

    # 5. Cálculo do EQM Final e Acurácia
    final_mse = som._compute_mse(data)
    accuracy = som.compute_accuracy(raw_data_norm, species_int)
    # Registrar no histórico da SOM para os métodos de plotagem
    som.history_mse.append(final_mse)
    som.history_accuracy.append(accuracy)
    print(f"Erro Quadrático Médio Final (EQM): {final_mse:.4f}")
    print(f"Acurácia Final da SOM: {accuracy * 100:.1f}%")

    # Gráficos de métricas via SOM
    som.plot_mse_history(save_path="src/iris_eqm.png")
    som.plot_accuracy_history(save_path="src/iris_accuracy.png")
    som.plot_neighborhood_decay(max_iterations=num_samples,
                                save_path="src/iris_neighborhood_decay.png")

    # 6. Executar K-Means e Plotar último estado
    print("Executando K-Means sobre os pesos finais (K=3)...")
    final_weights = np.array([node.neuron.weights.copy() for node in som.nodes])
    kmeans = KMeans(k=3)
    labels = kmeans.fit_predict(final_weights)
    
    # Plotar estado final com K-Means
    plot_final_kmeans(final_weights, labels, data, som.nodes, final_mse)

    # 7. Mapa de Grade da SOM com Clusters e Espécies Reais
    print("Plotando mapa de grade da SOM com clusters K-Means e espécies reais...")
    plot_cluster_grid(som, labels, species_int, raw_data_norm)

    # 8. Matriz de Confusão (via método nativo da SOM)
    print("Gerando matriz de confusão...")
    species_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    som.plot_confusion_matrix(
        data=raw_data_norm,
        true_labels=species_int,
        class_names=species_names,
        save_path="src/iris_confusion_matrix.png"
    )

if __name__ == "__main__":
    main()
