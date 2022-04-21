from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

class TfidfEncoder:
    @staticmethod
    def encode_vectors(data, tfidf_vectorizer):
        """
        Create a tfidf vectorizer.
        :param data: pandas dataframe (for training a model), utterance list (for running inference of trained model)
        :return: tfidf vectorized utterances
        """

        if isinstance(data, pd.DataFrame):
            data_df = data
            tfidf_utterance_vectors = tfidf_vectorizer.fit_transform(
                data_df['text'].values)

        elif isinstance(data, str):
            tfidf_utterance_vectors = tfidf_vectorizer.transform([data])

        return tfidf_utterance_vectors


    @staticmethod
    def encode_training_vectors(data_df):
        """
        Create a tfidf vectorizer.
        :param data_df: pandas dataframe (for training a model)
        :return: tfidf vectorized utterances
        """

        tfidf_vectorizer = TfidfVectorizer()

        tfidf_utterance_vectors = tfidf_vectorizer.fit_transform(
            data_df['text'].values)

        return tfidf_utterance_vectors, tfidf_vectorizer
