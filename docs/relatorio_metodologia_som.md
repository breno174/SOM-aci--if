# Metodologia: Classificação de Dados com Self-Organizing Map (SOM)

## 1. Introdução

Este documento descreve a metodologia utilizada para a classificação de dados utilizando uma **Rede Neural Auto-Organizável (Self-Organizing Map — SOM)**, implementada do zero em Python com NumPy. O objetivo é classificar amostras em três macro-classes de qualidade: **Ótimo**, **Normal** e **Ruim**, a partir de atributos numéricos extraídos dos dados originais.

A implementação inclui o treinamento da SOM, a rotulação dos neurônios por votação majoritária, a predição por meio dos *k* neurônios mais próximos (k-BMUs) com ponderação por distância, e a avaliação do modelo com múltiplas rodadas de treinamento para garantir robustez estatística dos resultados.

---

## 2. Pré-processamento dos Dados

### 2.1 Carregamento e Mapeamento de Classes

Os dados são carregados a partir de um arquivo CSV (`Dados.csv`) sem cabeçalho. A última coluna contém os rótulos originais das subclasses (por exemplo: `c1_p1`, `c2_p1`, `c3_p1`, `c3_p2`, `c4_p1`, `c4_p2`, `c4_p3`, `c4_p4`, etc.).

Esses rótulos são agrupados em **três macro-classes** segundo a seguinte regra:

| Macro-classe | Código | Subclasses originais            |
|--------------|--------|---------------------------------|
| Ótimo        | 0      | `c1_p1`                         |
| Normal       | 1      | `c2_p1`                         |
| Ruim         | 2      | `c3_*` e `c4_*` (todas as demais) |

### 2.2 Remoção de Colunas

Antes da normalização, três colunas são removidas do conjunto de atributos (colunas de índice 5, 12 e 13) por conterem informações irrelevantes ou redundantes para o treinamento.

### 2.3 Normalização (Z-score)

Os atributos numéricos são normalizados utilizando a padronização Z-score:

$$
x_{norm} = \frac{x - \mu}{\sigma}
$$

onde $\mu$ é a média e $\sigma$ é o desvio padrão de cada atributo, calculados sobre o conjunto completo de dados. Essa normalização garante que todos os atributos contribuam igualmente para o cálculo de distância Euclidiana na SOM.

---

## 3. Divisão Treino/Teste

### 3.1 Estratificação Hierárquica por Subclasse

A divisão dos dados entre treino e teste é feita de forma **estratificada por subclasse** (`y_raw`), não por macro-classe. Isso garante que cada subclasse individual (por exemplo, `c3_p2`, `c4_p4`) esteja representada proporcionalmente em ambos os conjuntos.

O procedimento é o seguinte:

1. Para cada subclasse única em `y_raw`:
   - Embaralha-se os índices das amostras dessa subclasse (com uma seed definida para reprodutibilidade).
   - Calcula-se o número de amostras de teste: `n_test = round(n × test_size)`.
   - Garante-se a presença mínima: se a subclasse possui 2 ou mais amostras, pelo menos 1 amostra vai para o teste e pelo menos 1 vai para o treino.
   - Se a subclasse possui apenas 1 amostra, ela é alocada inteiramente ao treino.
2. Os índices de treino e teste são embaralhados separadamente.

Essa abordagem é **genérica** e funciona para qualquer distribuição de subclasses, sem valores fixos ou codificados manualmente.

### 3.2 Proporção Utilizada

A proporção padrão utilizada é **80% treino / 20% teste** (`test_size=0.2`).

---

## 4. Arquitetura e Treinamento da SOM

### 4.1 Estrutura da Rede

A SOM é organizada como um grid bidimensional de neurônios com as seguintes características:

- **Grid**: M × N neurônios (configurável; exemplos testados incluem 4×4, 6×6, 7×7).
- **Dimensão**: cada neurônio possui um vetor de pesos com a mesma dimensionalidade dos dados de entrada.
- **Inicialização**: os pesos dos neurônios são inicializados aleatoriamente no intervalo [0, 1) utilizando `np.random.rand`.

### 4.2 Algoritmo de Treinamento

O treinamento segue o algoritmo clássico da SOM de Kohonen:

Para cada época *t* (de 1 a `NUM_EPOCHS`):
1. Os dados de treino são embaralhados aleatoriamente.
2. Para cada amostra $x$ do treino:
   - **Busca do BMU (Best Matching Unit)**: encontra-se o neurônio cujo vetor de pesos é mais próximo de $x$ pela distância Euclidiana.
   - **Atualização dos pesos**: todos os neurônios do grid são atualizados segundo a regra:

$$
w_i \leftarrow w_i + \alpha(t) \cdot h(bmu, i, t) \cdot (x - w_i)
$$

### 4.3 Função de Vizinhança Gaussiana

A influência de cada neurônio é determinada pela sua distância topológica ao BMU no grid:

$$
h(bmu, i, t) = \exp\left(-\frac{\|r_{bmu} - r_i\|^2}{2 \cdot \sigma(t)^2}\right)
$$

onde $r_{bmu}$ e $r_i$ são as posições (linha, coluna) do BMU e do neurônio *i* no grid, respectivamente. Essa função garante que neurônios vizinhos ao BMU sofram atualizações mais fortes que os mais distantes.

### 4.4 Decaimento dos Parâmetros

Os hiperparâmetros decaem ao longo do treinamento para garantir convergência:

- **Taxa de aprendizado** $\alpha(t)$: decai exponencialmente conforme:

$$
\alpha(t) = \alpha_0 \cdot \exp\left(-\frac{t}{T_{max}} \cdot 0.5\right)
$$

- **Raio de vizinhança** $\sigma(t)$: decai exponencialmente de $\sigma_0$ até $\sigma_{final} = 0.1$:

$$
\sigma(t) = \sigma_0 \cdot \left(\frac{\sigma_{final}}{\sigma_0}\right)^{t / T_{max}}
$$

onde $T_{max} = \text{NUM\_EPOCHS} \times N_{amostras}$ é o número total de iterações.

### 4.5 Hiperparâmetros

| Parâmetro              | Descrição                          | Valor (exemplo)  |
|------------------------|------------------------------------|------------------|
| `NUM_EPOCHS`           | Número de épocas de treinamento    | 150              |
| `GRID_M × GRID_N`     | Dimensões do grid de neurônios     | 4×4 a 7×7        |
| `LR` ($\alpha_0$)     | Taxa de aprendizado inicial        | 0.3              |
| `SIGMA` ($\sigma_0$)  | Raio de vizinhança inicial         | 3 a 7            |
| `K_BMU`               | Número de BMUs para predição       | 3                |

---

## 5. Rotulação dos Neurônios

### 5.1 Procedimento

Após o treinamento (fase não supervisionada), a SOM é transformada em um classificador por meio de uma etapa supervisionada de **rotulação dos neurônios**:

1. Para cada amostra do conjunto de **treino** $(x_i, y_{raw_i})$:
   - Encontra-se o BMU de $x_i$.
   - O rótulo `y_raw` da amostra é associado a esse neurônio.
2. Cada neurônio acumula uma lista de todos os rótulos `y_raw` que recebeu.
3. O rótulo final do neurônio é determinado por **votação majoritária** (*majority vote*): o rótulo que apareceu com maior frequência entre as amostras que ativaram aquele neurônio.

Neurônios que não foram ativados por nenhuma amostra de treino ficam sem rótulo (marcados como `None`).

### 5.2 Uso do `y_raw` (subclasses) ao invés do `y_macro`

A rotulação utiliza os rótulos originais das **subclasses** (`y_raw`, como `c3_p1`, `c4_p2`) e não os rótulos agrupados das macro-classes. Isso permite uma granularidade mais fina na representação interna da SOM e melhora a capacidade de discriminação no momento da predição.

---

## 6. Predição com k-BMUs Ponderados por Distância

### 6.1 Motivação

Diferentemente da abordagem clássica, onde a predição é feita considerando apenas o **1 BMU mais próximo**, esta implementação utiliza os **k neurônios mais próximos** (k-BMUs). Isso reduz a sensibilidade a neurônios individuais que possam estar mal rotulados e melhora a robustez do classificador.

### 6.2 Algoritmo de Predição

Para cada amostra de teste $x$:

1. Calcula-se a distância Euclidiana de $x$ para **todos** os neurônios do grid.
2. Seleciona-se os **k** neurônios com menor distância (k-BMUs).
3. Para cada um dos k neurônios selecionados que possuem rótulo, calcula-se um peso inversamente proporcional à distância:

$$
w_j = \frac{1}{d(x, n_j) + \varepsilon}
$$

onde $\varepsilon = 10^{-8}$ evita divisão por zero.

4. Os votos ponderados são acumulados por rótulo `y_raw`:

$$
\text{voto}(c) = \sum_{j \in k\text{-BMUs},\ label_j = c} w_j
$$

5. A classe predita é o `y_raw` com maior soma de votos ponderados.

### 6.3 Conversão para Macro-Classe

Após a predição em nível de subclasse (`y_raw_pred`), os rótulos são convertidos para macro-classes conforme o mapeamento:

- `c1_p1` → **Ótimo (0)**
- `c2_p1` → **Normal (1)**
- Qualquer outro rótulo → **Ruim (2)**

---

## 7. Avaliação com Múltiplas Rodadas

### 7.1 Protocolo de Avaliação

Para garantir a robustez estatística dos resultados, o processo completo de treino e teste é executado **múltiplas vezes** (`NUM_RUNS` rodadas), cada uma com uma seed aleatória diferente. Isso resulta em:

- Diferentes divisões treino/teste a cada rodada.
- Diferentes inicializações de pesos e ordens de apresentação das amostras.

### 7.2 Métricas Coletadas por Rodada

| Métrica                   | Descrição                                                            |
|---------------------------|----------------------------------------------------------------------|
| Acurácia de Treino        | Proporção de acertos no conjunto de treino (via k-BMU)               |
| Acurácia de Teste         | Proporção de acertos no conjunto de teste (via k-BMU)                |
| EQM por Época (Treino)    | Erro Quadrático Médio (Quantization Error) a cada época no treino    |
| EQM por Época (Teste)     | EQM calculado retroativamente com os snapshots de pesos em cada época|
| Matriz de Confusão Treino | Matriz de confusão acumulada das predições no treino                 |
| Matriz de Confusão Teste  | Matriz de confusão acumulada das predições no teste                  |

### 7.3 Erro Quadrático Médio (EQM)

O EQM (ou *Quantization Error*) mede a qualidade da representação topológica da SOM:

$$
EQM = \frac{1}{N} \sum_{i=1}^{N} \|x_i - w_{bmu(x_i)}\|^2
$$

É a média das distâncias ao quadrado entre cada amostra e o vetor de pesos do seu BMU. Um EQM menor indica que os neurônios da SOM representam melhor os dados.

O EQM de teste é calculado **retroativamente** ao final de cada rodada: para cada época, utiliza-se o snapshot dos pesos armazenados durante o treino para computar o EQM sobre o conjunto de teste, sem qualquer vazamento de dados.

---

## 8. Visualizações Geradas

### 8.1 Gráfico de Acurácia por Rodada

Exibe a acurácia de treino e teste para cada rodada de treinamento, com linhas horizontais indicando as médias gerais. Permite avaliar a variabilidade dos resultados entre diferentes execuções.

### 8.2 Evolução do EQM (Treino e Teste)

Dois gráficos lado a lado exibindo:

- **Esquerda**: Evolução do EQM médio de **treino** ao longo das épocas (média de todas as rodadas), com faixa de ±1 desvio padrão.
- **Direita**: Evolução do EQM médio de **teste** ao longo das épocas (média de todas as rodadas), com faixa de ±1 desvio padrão.

Espera-se que ambos apresentem uma tendência de queda, indicando a convergência da SOM.

### 8.3 Matrizes de Confusão Acumuladas

Duas matrizes de confusão lado a lado:

- **Esquerda**: Matriz de confusão acumulada do **treino** (soma de todas as rodadas).
- **Direita**: Matriz de confusão acumulada do **teste** (soma de todas as rodadas).

Cada célula indica quantas vezes uma amostra de classe real *i* foi predita como classe *j*, somando-se todas as rodadas. A acurácia global acumulada é exibida no título de cada matriz.

---

## 9. Reprodutibilidade

Cada rodada de treinamento é salva em um arquivo JSON contendo:

- A **seed** aleatória utilizada.
- Os **pesos iniciais** dos neurônios.
- A **configuração completa** dos hiperparâmetros (grid, taxa de aprendizado, sigma, número de épocas, k-BMU).
- A **acurácia** de treino e teste daquela rodada.

Isso permite reproduzir exatamente qualquer experimento futuro ao carregar o arquivo e recriar a SOM com os mesmos parâmetros.

---

## 10. Resumo do Pipeline

```
Dados CSV
   │
   ├── Remoção de colunas irrelevantes
   ├── Mapeamento de subclasses → macro-classes
   └── Normalização Z-score
         │
         ▼
  Divisão Treino/Teste
  (estratificada por subclasse, 80/20)
         │
         ▼
  Treinamento da SOM (N rodadas)
  ┌──────────────────────────────────────┐
  │ Para cada rodada:                    │
  │  1. Inicializar SOM                  │
  │  2. Treinar por NUM_EPOCHS           │
  │  3. Rotular neurônios (y_raw)        │
  │  4. Predizer com k-BMUs ponderados   │
  │  5. Converter y_raw → y_macro        │
  │  6. Coletar métricas                 │
  │  7. Salvar experimento (.json)       │
  └──────────────────────────────────────┘
         │
         ▼
  Geração de Gráficos Agregados
  (Acurácia, EQM, Matrizes de Confusão)
```
