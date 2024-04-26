# Sentiment Analysis of IMDB Movie reviews
* The model uses 2 algorithm to find out about whether a given review is Positive or Negative
* The model score is 89% for LogisticRegression() and RandomForestClassifier()
Sentiment analysis AI models are designed to analyze and understand the sentiment or opinion expressed in a piece of text. These models can determine whether the sentiment conveyed in the text is positive, negative, or neutral. Here's an overview of how sentiment analysis AI models work:

1. **Text Preprocessing**: The input text undergoes preprocessing steps such as tokenization (breaking text into individual words or tokens), removing stopwords (commonly used words that carry little meaning), and stemming/lemmatization (reducing words to their base form).

2. **Feature Extraction**: After preprocessing, the text is transformed into a numerical representation suitable for input into machine learning models. Common techniques include bag-of-words (BoW), term frequency-inverse document frequency (TF-IDF), or word embeddings (e.g., Word2Vec, GloVe).

3. **Model Selection**: There are various machine learning and deep learning models used for sentiment analysis, including:

   - **Naive Bayes**: A probabilistic model based on Bayes' theorem, commonly used for text classification tasks.
   - **Support Vector Machines (SVM)**: A supervised learning model that separates data points into different classes using hyperplanes.
   - **Logistic Regression**: A linear model that predicts the probability of a binary outcome.
   - **Recurrent Neural Networks (RNNs)**: Deep learning models designed to handle sequential data, such as text. Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) are popular variants of RNNs used for sentiment analysis.
   - **Convolutional Neural Networks (CNNs)**: Deep learning models commonly used for image analysis but can also be applied to text classification tasks by treating text as one-dimensional data.

4. **Training**: The selected model is trained on labeled datasets containing examples of text along with their corresponding sentiment labels (positive, negative, or neutral). During training, the model learns to identify patterns and features in the text that are indicative of sentiment.

5. **Evaluation and Fine-tuning**: After training, the model is evaluated on a separate validation dataset to assess its performance. Metrics such as accuracy, precision, recall, and F1-score are used to evaluate the model's effectiveness. The model may be fine-tuned by adjusting hyperparameters or incorporating additional training data to improve performance.

6. **Deployment**: Once the model achieves satisfactory performance, it can be deployed in real-world applications to analyze the sentiment of user-generated content, social media posts, customer reviews, and other text data sources.

Popular libraries and frameworks for building sentiment analysis AI models include TensorFlow, PyTorch, Scikit-learn, NLTK (Natural Language Toolkit), and spaCy.

Overall, sentiment analysis AI models play a crucial role in understanding and extracting insights from text data, enabling businesses to make informed decisions based on customer feedback, market sentiment, and social media trends.
