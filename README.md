# ml-confusion-matrix-12-24
Este script é um exemplo de desenvolvimento de um modelo de machine learning para reconhecimento de dígitos escritos à mão usando o conjunto de dados MNIST, com funções adicionais para avaliação de desempenho do modelo.

O código pode ser dividido em várias seções principais:

## 1. Importação de Bibliotecas
```python
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import seaborn as sns
import pandas as pd
from tensorflow.keras import datasets, layers, models
```
Aqui são importadas bibliotecas essenciais para:
- Visualização de dados (matplotlib, seaborn)
- Processamento de dados (numpy, pandas)
- Desenvolvimento de machine learning (tensorflow, keras)

## 2. Carregamento e Preparação dos Dados
```python
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

train_images, test_images = train_images / 255.0, test_images / 255.0
```
- Carrega o dataset MNIST (imagens de dígitos escritos à mão)
- Reshape das imagens para o formato de entrada da rede neural
- Normalização dos pixels para o intervalo 0-1

## 3. Definição do Modelo de Rede Neural Convolucional
```python
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
```
Arquitetura da rede neural:
- Três camadas convolucionais para extração de características
- Camadas de pooling para redução dimensional
- Camada flatten para transformar em vetor
- Duas camadas densas finais para classificação
- Última camada com 10 neurônios (para 10 dígitos) e softmax para probabilidades

## 4. Compilação e Treinamento do Modelo
```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x=train_images, y=train_labels, epochs=5, validation_data=(test_images, test_labels))
```
- Usa otimizador Adam
- Função de perda: sparse categorical crossentropy
- Treina por 5 épocas

## 5. Funções de Avaliação de Modelo
O script inclui várias funções importantes para análise de desempenho:

a) `calcular_especificidade()`: Calcula a especificidade a partir da matriz de confusão

b) `plotar_matriz_confusao()`: Cria uma visualização heatmap da matriz de confusão normalizada

c) `plotar_curva_roc()`: Plota a curva ROC (Receiver Operating Characteristic) e calcula a área sob a curva (AUC)

d) `avaliar_classificacao()`: Função principal que calcula e exibe múltiplas métricas:
- Acurácia
- Precisão
- Sensibilidade (Recall)
- Especificidade
- F1-Score

## 6. Visualização da Matriz de Confusão
```python
y_true = test_labels
y_pred = np.argmax(model.predict(test_images), axis=-1)

con_mat = tf.math.confusion_matrix(labels=y_true, predictions=y_pred).numpy()
con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

figure = plt.figure(figsize=(8, 8))
sns.heatmap(con_mat_norm, annot=True, cmap=plt.cm.Blues)
```
Cria uma matriz de confusão normalizada e a visualiza como um heatmap

## 7. Exemplo de Uso
Na seção `if __name__ == "__main__"`, há um exemplo de uso das funções de avaliação com dados fictícios de classificação binária.

Pontos interessantes:
- O código demonstra um fluxo completo de machine learning
- Inclui técnicas de pré-processamento de dados
- Usa uma arquitetura de rede neural convolucional
- Fornece múltiplas métricas para avaliação de desempenho
- Oferece visualizações para melhor interpretação dos resultados
