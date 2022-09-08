# NLP Python Projects
This is a series of projects for the Winter 2021 Artificial Intelligence II course on [DIT@UoA](https://www.di.uoa.gr/en).

## Goals & Tasks
The course is focused on **Natural Language Processing**. The main areas of interest of this project series are:
- **Vaccine Sentiment Classification**
- **Question Answering** on the [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) dataset, by fine-tuning the `BertForQuestionAnswering` pretrained model
- The thoretical background of the tools used throughout the project
- Familiarizing with the development procedure of ML models
- Experimenting with a variety of hyperparameters, comparing across different architectures & designs, measuring and reporting the results.

## Technologies & Tools used for development
- **scikit-learn**
- **PyTorch**
- VS Code, Google Colab & Kaggle

## Data & Dependencies
- The Classification tasks are conducted on **Tweets**, to determine whether the author is **0. Neutral**, **1. Anti-Vaccine** or **2. Pro-Vaccine**. Train & Validation sets can be found on `datasets`.

- Projects 2 & 3 use the pretrained [GloVe word embeddings](https://nlp.stanford.edu/projects/glove/). The selected archive is `glove.6.b.zip`. To use them, download the archive and extract it in `embeddings`.

- Project 4 uses the SQuAD 2.0 [train](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json) & [dev](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json) sets, which can be downloaded & placed in `embeddings` as well.

## Projects Summary
- **Project 1**:\
Vaccine Sentiment Classifier using **Softmax Regression**, implemented with `scikit-learn`.
    - Measuring with Precision, Recall, F1 Score & Confusion Matrices
- **Project 2**:\
Vaccine Sentiment Classifier using **Feed-Forward Neural Network** & **pre-trained GloVe word embedding vectors**, implemented in `PyTorch`.
    - Experimenting with hyperparameters
    - Measuring with Precision, Recall, F1 Score, Confusion Matrices, Loss-Epochs & ROC Curves
    - Comparison with Project 1 model
- **Project 3**:\
Vaccine Sentiment Classifier using **Bidirectional, Stacked LSTM/GRU Recursive Neural Networks** & **pre-trained GloVe word embedding vectors**, implemented in `PyTorch`.
    - Experimenting with number of hidden layers, LSTM/GRU cells, skip connections & other hyperparameters
    - Measuring with Precision, Recall, F1 Score, Confusion Matrices, Loss-Epochs & ROC Curves
    - Comparison with Projects 1 & 2 models
- **Project 4**:
    - Vaccine Sentiment Classifier using a **fine-tuned BERT-base model** from [Hugging Face](https://huggingface.co/models)
        - Measuring with Precision, Recall, F1 Score, Confusion Matrices & ROC Curve
    - Question Answering model for **SQuAD 2.0**, using a **fine-tuned BERT-base model**

All projects include the following:
- A `task.pdf` file describing the corresponding project tasks
- A `README.md` file which includes the experiment presentation, reports & comparisons
- `.ipynb` files with the implemented models
- A `docs` directory with the solutions for any theoretical tasks.
