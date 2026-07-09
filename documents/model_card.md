Este **Model Card** foi estruturado para documentar o projeto de sistema de recomendação, servindo como guia técnico e de transparência para o modelo `Neural-NeuMF-MLP`.

---

# Model Card: Neural-NeuMF-MLP Recommender

## 1. Model Details

* **Nome do Modelo:** Neural-NeuMF-MLP (Neural Collaborative Filtering).
* **Versão:** 1.0.
* **Tipo de Modelo:** Sistema de Recomendação Híbrido (Matrix Factorization + MLP).
* **Arquitetura:** Combinação de Fatoração de Matrizes (GMF) e Perceptron Multicamadas (MLP).
* **Engenheira do projeto:** Lara Gonçalves da Silva.

## 2. Intended Use

* **Objetivo:** Prever a afinidade de usuários por itens específicos da plataforma RetailRocket.
* **Aplicação:** Personalização de recomendações em tempo real para aumentar o engajamento e a taxa de cliques (CTR).
* **Usuários pretendidos:** Sistemas de recomendação em e-commerce.

## 3. Factors & Quantitative Analysis (Dataset)

* **Dataset:** RetailRocket (Events and Properties).
* **Análise Quantitativa:**
* **Volume:** [Inserir qtd total de eventos].
* **User/Item Sparsity:** O dataset apresenta alta dispersão, característica comum em logs de navegação de e-commerce.
* **Distribuição:** Eventos majoritariamente de "view", com menor incidência de "add-to-cart" e "transaction".


* **Factors:** O desempenho pode variar dependendo da atividade do usuário (usuários novos com poucas interações apresentam performance reduzida).

## 4. Metrics & Evaluation

* **Métricas de Avaliação:**
* **Precision@10:** Proporção de itens relevantes no top 10.
* **Recall@10:** Capacidade de recuperar itens relevantes.
* **NDCG@10:** Qualidade do ranqueamento (considerando a ordem).
* **Catalog Coverage:** Proporção de itens do catálogo que o modelo é capaz de recomendar.


* **Evaluation Data:** Subconjunto de teste com dados segregados por `timestamp` para evitar *data leakage*.

---

## 5. Fluxo de Trabalho (End-to-End)

1. **Ingestão:** Leitura dos arquivos brutos do RetailRocket.
2. **Preprocessing:** Mapeamento de `user_id` e `item_id` para índices contínuos.
3. **Treinamento:** Otimização via Adam usando `MSELoss` (ou `BCEWithLogitsLoss`).
4. **Avaliação:** Cálculo das métricas de ranqueamento via `AvaliarSistemaRecomendacao`.
5. **Log:** Registro de métricas e artefatos no MLflow.

## 6. Organização do Repositório

```text
├── shared/             # Utilitários, Preprocessamento e Modelos
├── training/           # Scripts de treino (train.py)
├── docker-compose.yml  # Orquestração (MLflow + App)
└── mlruns/             # Persistência de experimentos

```

## 7. Guia de Execução e Docker

### Execução via Docker

Para subir o ambiente completo (MLflow + App):

```bash
docker-compose up -d --build

```

Para executar o treino dentro do container:

```bash
docker-compose exec recommender_app python training/train.py

```

## 8. Endpoints e Exemplos de Retorno (MLflow Serving)

Ao servir o modelo (`mlflow models serve`), o endpoint `/invocations` espera:

* **Input:** `{"inputs": [[user_idx], [item_idx]]}`
* **Output:** `{"predictions": [score_value]}`

## 9. Testes e Qualidade

* **Unit Tests:** Validam os mapeamentos do preprocessor.
* **Integração:** Validam a conexão com o MLflow (Tracking URI).
* **Dataset Integrity:** Verificação de tipos (`int64` vs `float32`) para evitar *schema enforcement errors*.

## 10. Execução da Análise Exploratória (EDA)

A EDA está contida no diretório `notebooks/`. Recomenda-se executar o `eda.ipynb` para visualizar a distribuição de eventos e a cauda longa (*long tail*) dos produtos.

## 11. Resultados dos Experimentos

* **Loss de Treino:** Convergência observada em ~3 épocas.
* **Performance Final:**
* `Precision@10`: [Inserir valor]
* `Recall@10`: [Inserir valor]
* `NDCG@10`: [Inserir valor]

