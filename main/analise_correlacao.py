import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# Adiciona o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_and_prepare_data():
    print("[1] Carregando dados...")
    df = pd.read_csv("data/Dados.csv", header=None)

    y_raw = df.iloc[:, -1]

    # ── Agrupar em 3 macro-classes como feito no treinamento ───────
    def _map_to_group(label: str) -> int:
        if label.startswith("c1"):
            return 0  # Ótimo
        elif label.startswith("c2"):
            return 1  # Normal
        else:
            return 2  # Ruim

    y = y_raw.map(_map_to_group).values

    # Remover colunas indesejadas + label (exatamente como no aco_multi_treino.py)
    cols_to_drop = [5, 12, 13]
    # Guardar os nomes (ou índices) originais das features para os gráficos
    original_cols = list(range(df.shape[1] - 1))
    valid_cols = [c for c in original_cols if c not in cols_to_drop]
    
    X_df = df[valid_cols]
    
    # Renomear as colunas para "Feature X" para facilitar a visualização
    feature_names = [f"F_{c}" for c in valid_cols]
    X_df.columns = feature_names
    
    return X_df, y, feature_names

def plot_correlation_matrix(X_df, save_dir):
    print("[2] Calculando e plotando Matriz de Correlação (Features x Features)...")
    corr_matrix = X_df.corr(method='pearson')

    n_features = len(corr_matrix)
    cell_size = 0.55  # tamanho de cada célula em polegadas
    fig_size = max(6, n_features * cell_size)

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        linecolor='white',
        annot_kws={"size": 7, "weight": "bold"},
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )

    # ── Ajustar cor do texto para alto contraste ──
    # Valores extremos (azul/vermelho escuro) → texto branco
    # Valores próximos de 0 (fundo claro)      → texto preto
    for text_obj in ax.texts:
        val = float(text_obj.get_text())
        text_obj.set_color("white" if abs(val) > 0.45 else "black")

    ax.set_title(
        "Matriz de Correlação de Pearson entre as Features",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    plt.tight_layout()

    save_path = os.path.join(save_dir, "matriz_correlacao_features.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   Gráfico salvo em: {save_path}")
    plt.close()

def analyze_target_influence(X_df, y, feature_names, save_dir):
    print("[3] Analisando influência das features no alvo (Macro-classes)...")
    
    # Adiciona o target temporariamente para correlação
    df_with_target = X_df.copy()
    df_with_target['Target'] = y
    
    # 1. Correlação Linear (Pearson e Spearman com o Target)
    corr_with_target = df_with_target.corr(method='pearson')['Target'].drop('Target')
    corr_spearman = df_with_target.corr(method='spearman')['Target'].drop('Target')
    
    # 2. Importância usando Random Forest (para capturar relações não-lineares)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_df, y)
    importances = rf.feature_importances_
    
    # Consolida os resultados num DataFrame
    influence_df = pd.DataFrame({
        'Feature': feature_names,
        'Corr_Pearson': corr_with_target.values,
        'Corr_Pearson_Abs': np.abs(corr_with_target.values),
        'Corr_Spearman': corr_spearman.values,
        'Corr_Spearman_Abs': np.abs(corr_spearman.values),
        'RF_Importance': importances
    })
    
    # Ordenar pela importância do Random Forest
    influence_df = influence_df.sort_values(by='RF_Importance', ascending=False)
    
    print("\nRanking de Influência (Ordenado por Importância do Random Forest):")
    print(influence_df[['Feature', 'RF_Importance', 'Corr_Pearson', 'Corr_Spearman']].to_string(index=False))
    
    # Salvar resultados em CSV
    csv_path = os.path.join(save_dir, "influencia_features.csv")
    influence_df.to_csv(csv_path, index=False)
    
    # Plotar Importância do Random Forest
    plt.figure(figsize=(12, 6))
    sns.barplot(data=influence_df, x='Feature', y='RF_Importance', palette='viridis')
    plt.title("Importância das Features para o Treino (Random Forest)", fontsize=14, fontweight='bold')
    plt.ylabel("Importância Relativa")
    plt.xlabel("Features")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    rf_plot_path = os.path.join(save_dir, "importancia_random_forest.png")
    plt.savefig(rf_plot_path, dpi=150, bbox_inches='tight')
    print(f"   Gráfico salvo em: {rf_plot_path}")
    plt.close()
    
    # Plotar Correlação de Pearson Absoluta
    influence_df_pearson = influence_df.sort_values(by='Corr_Pearson_Abs', ascending=False)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=influence_df_pearson, x='Feature', y='Corr_Pearson_Abs', palette='coolwarm')
    plt.title("Correlação Absoluta (Pearson) das Features com o Alvo", fontsize=14, fontweight='bold')
    plt.ylabel("Correlação Absoluta (|r|)")
    plt.xlabel("Features")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    pearson_plot_path = os.path.join(save_dir, "correlacao_absoluta_pearson.png")
    plt.savefig(pearson_plot_path, dpi=150, bbox_inches='tight')
    print(f"   Gráfico salvo em: {pearson_plot_path}")
    plt.close()

def main():
    save_dir = "src/acoPictures/correlacao"
    os.makedirs(save_dir, exist_ok=True)
    
    X_df, y, feature_names = load_and_prepare_data()
    
    plot_correlation_matrix(X_df, save_dir)
    
    analyze_target_influence(X_df, y, feature_names, save_dir)
    
    print("\n✅ Análise de correlação e influência concluída!")

if __name__ == "__main__":
    main()
