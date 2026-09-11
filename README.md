# Applied Data Science for Greek Legal Documents

Collection of notebooks of Data Extraction, Machine learning (Classification/ Clustering), and LLM-assisted analysis applied to Greek legal text.
This repository explores how unstructured and highly specialized legal documents can be collected, transformed into analysis-ready data, and examined using both classical machine-learning methods and modern language-model workflows. The projects cover the full path from web crawling and exploratory analysis to large-scale multiclass classification, semantic clustering, and few-shot interpretation.

# Legal Docs Challenge

Greek legal documents present a demanding real-world data problem: the text is domain-specific, labels are severely imbalanced, and hundreds or even thousands of narrowly defined categories may coexist. Rather than presenting a single model in isolation, this repository compares multiple representations and ml algorithms, evaluates which methods are useful for different tasks and derives useful insights about legal texts.

## Project overview

### 1. Greek Supreme Court Decision Collection

A Selenium-based crawler collects 2024 civil and criminal decisions from the **Greek Supreme Court**. It converts semi-structured web pages to a format comparable to Hugging Face Greek legal dataset.
The workflow includes regex expressions which extract potential useful information such as decision number and year, department ,judges, Introductory text and articles mentioned in the case
Referenced articles of Greek legislation, explicit waits for dynamically loaded elements, logging, error handling, and exports to JSON Lines and CSV.


### 3. Large-Scale Legal Document Classification

The goal of this notebook is classification whilst using traditional text representations and dense embeddings across three chapters: **Volume**, **Chapter**, and **Subject** of an already existing Huggings' Face dataset.

This project addresses a particularly difficult classification problem because the dataset combines severe class imbalance with a very large number of unique labels. A small number of categories contain most documents, while many classes have very few training examples.

The difficulty increases across the chapters. Subject is the most challenging task, containing more than 2,000 unique classes. 

#### Text representations:####  Bag of Words / TF-IDF / Corpus-trained Word2Vec embeddings / Pretrained Greek fastText embeddings

#### Classifiers:#### Logistic Regression / Support Vector Machines / XGBoost

**Results:** The strongest and most consistent approach was **Logistic Regression with TF-IDF**. It achieved approximately:

* **80% accuracy for Volume**
* **73.5% accuracy for Chapter**
* **60% accuracy for Subject**

### 2. Exploratory Analysis and Clustering of Legal Text

The project aims to get the best clustering, compares K-Means with Word2Vec embeddings, BERTopic pipeline using multilingual sentence-transformer embeddings, UMAP, and HDBSCAN.

The BERTopic and HDBSCAN pipeline identified **168 non-noise clusters** and achieved better separation by allowing irregular cluster shapes and leaving uncertain documents unassigned. Silhouette score at 

### 5. Few-Shot Cluster Interpretation with Llama

The final project uses three-shot prompting to generate concise thematic descriptions for selected clusters.

Two example-selection strategies are compared:

* Documents nearest to each cluster centroid
* Randomly selected documents from the same cluster

The generated descriptions are compared with the most frequent known titles in each cluster. This analysis explores how the selection of representative examples affects LLM interpretation and connects traditional clustering with modern generative-AI workflows.


## Technologies

`Python` · `beautifulsoup` . `pandas` · `NumPy` · `scikit-learn` · `Selenium` · `XGBoost` · `gensim` · `fastText` · `SentenceTransformers` · `UMAP` · `HDBSCAN` · `BERTopic` · `Matplotlib` · `Llama LLM`

## Repository Organization

The notebooks are grouped according to the stage of the analytical workflow:

```text
├── crawling_scraping.ipynb          # Areios Pagos crawling and extraction
├── embeddings_classification.ipynb  # text representations and classification models
├── analysis_clustering.ipynb        # EDS, K-Means, BERTopic and HDBSCAN analysis
└── few_shot_learn.ipynb             # Llama-based few-shot cluster interpretation
```

> The crawling and modelling notebooks use complementary Greek legal sources and should be treated as related portfolio projects rather than as stages operating on one shared dataset.


