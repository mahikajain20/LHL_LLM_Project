# LLM Project - Mahika Jain

## Sentiment Analysis Model

I used Google Colab for this exercise, therefore I had trouble uploading the datasets and models to the GitHub. 
However, they can be accessed using the code in the notebooks and links provided.

## Project Task
- Task: Sentiment analysis on movie reviews
- Goal: Classify movie reviews as positive or negative sentiment
- Application: Automated sentiment analysis for film industry feedback

## Dataset
- Name: IMDB Movie Reviews Dataset
- Size: 50,000 reviews (25,000 for training, 25,000 for testing)
- Features: Text of movie reviews and corresponding sentiment labels
- Source: [https://huggingface.co/datasets/stanfordnlp/imdb]
- Preprocessing: Removed HTML tags, lowercased text, tokenized reviews

## Pre-trained Model
- Models: SiEBERT - English-Language Sentiment Classification (https://huggingface.co/siebert/sentiment-roberta-large-english)
- Model 2 was DistilBERT base uncased (https://huggingface.co/distilbert/distilbert-base-uncased)
- Architecture: Transformer-based language model
- It enables reliable binary sentiment analysis for various types of English-language text.
- Reason for selection: Efficient balance of performance and computational requirements

## Performance Metrics
- Primary metric: Accuracy
- Secondary metrics: F1 score, Precision, Recall
- Results:
  - Accuracy: 87.42%

## Hyperparameters
Key hyperparameters and their values:
- Learning rate: 2e-5
- Batch size: 16
- Number of epochs: 1
- Max sequence length: 256
- Optimizer: AdamW
- Weight decay: 0.01

Most impactful hyperparameters:
1. Learning rate: Crucial for model convergence
2. Batch size: Affected training speed and stability
3. Number of epochs: Balanced between underfitting and overfitting

## Final Model 

Model - [https://huggingface.co/Mahikajain/my_model]

