# 🎭 Reconhecimento de Faces com PCA e SVM

> **Disciplina:** Inteligência Artificial — PUC Goiás  
> **Aluno:** Hakawã Luiz Bernardi

---

## 📌 Objetivo Geral

Desenvolver um sistema de reconhecimento facial utilizando técnicas de Aprendizado de Máquina, mais especificamente a combinação de **Análise de Componentes Principais (PCA)** com **Máquinas de Vetores de Suporte (SVM)**, aplicadas à base de dados **ORL (Olivetti Research Laboratory)**.

---

## 🗃️ Base de Dados ORL

A base ORL (*Olivetti Research Laboratory*) contém:

- **40 sujeitos distintos**, cada um com **10 imagens**
- Total de **400 imagens** em formato `.pgm`
- Imagens capturadas entre 1992 e 1994 no laboratório Olivetti em Cambridge, UK
- Resolução: **92×112 pixels**, escala de cinza (8 bits)
- Variações incluem: iluminação, expressões faciais (olhos abertos/fechados, sorrindo/neutro) e detalhes faciais (óculos/sem óculos)

---

## ⚙️ Pipeline do Sistema

### 1. Leitura e Pré-processamento
- Carregamento das 400 imagens da base ORL
- Conversão para escala de cinza
- Redimensionamento para o tamanho padrão **112×92 pixels**
- Vetorização: cada imagem se torna um vetor de 10.304 dimensões

### 2. Redução de Dimensionalidade — PCA
- Aplicação de **PCA com 50 componentes principais**
- Redução de 10.304 → 50 dimensões
- Preservação das principais variâncias da face (eigenfaces)

### 3. Classificação — SVM
- Treinamento de um **classificador SVM com kernel RBF**
- Entrada: vetores PCA de 50 dimensões
- Saída: identidade do sujeito (classe 1–40)

### 4. Teste e Visualização
- Seleção aleatória de uma imagem de teste
- Predição da classe e exibição das 9 imagens da classe prevista
- Exibição do **Top-5 classes mais prováveis** com respectivas confianças
- **Matriz de confusão** com validação cruzada 5-fold
- **Projeção t-SNE** dos vetores PCA, coloridos por classe

---

## 📊 Saídas Geradas

| Saída | Descrição |
|-------|-----------|
| Imagem de teste | Imagem selecionada com a classe prevista |
| Grade 2×5 | Imagem de teste + 9 imagens da classe prevista |
| Top-5 classes | As 5 classes mais prováveis e suas confianças (%) |
| Matriz de confusão | Avaliação com validação cruzada estratificada 5-fold |
| Projeção t-SNE | Visualização 2D da separabilidade entre sujeitos |

---

## 🚀 Como Executar

### Pré-requisitos

```bash
pip install numpy matplotlib pillow scikit-learn seaborn
```

### Executando

1. Clone ou baixe este repositório
2. Ajuste o caminho da base no script:

```python
# Em reconhecimentoFace.py, linha ~8
base_path = r"./orl_faces"  # caminho relativo ao repositório
```

3. Execute:

```bash
python reconhecimentoFace.py
```

---

## 📈 Resultados

O modelo apresenta alta acurácia na base ORL, com a maioria das classes classificadas corretamente conforme evidenciado na matriz de confusão (validação cruzada 5-fold). O t-SNE demonstra boa separabilidade entre os sujeitos no espaço PCA reduzido.

Erros ocorrem principalmente entre sujeitos com características faciais similares ou imagens com variações mais extremas de iluminação e expressão.

---

## 📚 Referências

- Kitani, E. C., Thomaz, C. E. *Um Tutorial sobre Análise de Componentes Principais para o Reconhecimento Automático de Faces*. Relatório Técnico, Departamento de Engenharia Elétrica, Centro Universitário da FEI.
- Samaria, F. e Harter, A. *"Parameterisation of a stochastic model for human face identification"*, 2nd IEEE Workshop on Applications of Computer Vision, Dezembro 1994, Sarasota (Florida).

---

## 📄 Licença

Base de dados ORL: créditos ao **Olivetti Research Laboratory**, Cambridge, UK.
