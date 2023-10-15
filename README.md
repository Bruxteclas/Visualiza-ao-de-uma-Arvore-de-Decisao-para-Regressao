## Exercício 2: Visualização de uma Árvore de Decisão para Regressão

Neste exercício, realizamos a visualização de uma árvore de decisão para problemas de regressão. Abaixo estão as etapas e os códigos utilizados para completar o exercício:

### Passo 1: Carregar o Conjunto de Dados

```python
import pandas as pd

# URL do arquivo CSV
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
data = pd.read_csv(url)
```

### Passo 2: Preparar os Dados

```python
# Definir as features (variáveis independentes) e a coluna alvo (variável dependente)
X = data.drop('medv', axis=1)  # Excluir a coluna 'medv' do conjunto de features
y = data['medv']  # Usar a coluna 'medv' como a variável alvo

# Dividir o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Passo 3: Treinar uma Árvore de Regressão

```python
from sklearn.tree import DecisionTreeRegressor

# Criar uma árvore de decisão com profundidade máxima 5
regressor = DecisionTreeRegressor(max_depth=5)
regressor.fit(X_train, y_train)
```

### Passo 4: Calcular o Caminho Indicado pelos CCP-Alfas

Este passo pode ser realizado usando a função `cost_complexity_pruning_path`.

### Passo 5: Treinar Árvores com Alfas e Calcular o MSE

Neste passo, você percorrerá os valores de alfa e treinará uma árvore para cada alfa, calculando o MSE para cada uma delas.

### Passo 6: Visualizar a Árvore Encontrada

Por fim, visualize a árvore de decisão resultante após escolher um valor de alfa perto do ponto de mínimo do MSE.

Lembre-se de ajustar os códigos e adicionar a parte de cálculo do caminho e treinamento com base nos valores de alfa (Passos 4 e 5). Para calcular o R-quadrado (Passo 6), você pode usar métricas de avaliação como o R-quadrado fornecido pelo Scikit-Learn.

Esse resumo apresenta uma visão geral das etapas do Exercício 2 e pode ser usado para documentar seu trabalho no repositório. Certifique-se de adicionar os códigos e detalhes específicos para cada passo.
