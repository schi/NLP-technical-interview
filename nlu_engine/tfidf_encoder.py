from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class TfidfEncoder:
    @staticmethod
    def get_vectors(text_data, tfidf_vectorizer):
        """
        Get tfidf vectors.
        :param text_data: pandas dataframe column of text
        :param tfidf_vectorizer: tfidf vectorizer
        :return: tfidf vectorized text
        """
        tfidf_utterance_vectors = tfidf_vectorizer.transform(text_data.values)

        return tfidf_utterance_vectors

    @staticmethod
    def get_vectors_from_string(string, tfidf_vectorizer):
        """
        Get tfidf vectors.
        :param string: string
        :param tfidf_vectorizer: tfidf vectorizer
        :return: tfidf vectorized string
        """
        tfidf_utterance_vectors = tfidf_vectorizer.transform([string])

        return tfidf_utterance_vectors


    @staticmethod
    def create_vectorizer(text_data):
        """
        Create a tfidf vectorizer and get the vectors of the text.
        :param text_data: pandas dataframe column of text
        :return: tfidf vectorized utterances and the vectorizer
        """

        tfidf_vectorizer = TfidfVectorizer()

        tfidf_utterance_vectors = tfidf_vectorizer.fit_transform(
            text_data.values)

        return tfidf_utterance_vectors, tfidf_vectorizer


    @staticmethod
    def get_top_n_features(tfidf_vectorizer, tfidf_vectors, n):
        """
        Get the top n features from text data with the tfidf vectorizer.
        :param tfidf_vectorizer: tfidf vectorizer
        :param n: number of features to return
        :return: top n features
        """
        
        feature_names = np.array(tfidf_vectorizer.get_feature_names())
        sorted_features = np.argsort(tfidf_vectors.data)[:-(n+1):-1]
        top_n_features = feature_names[tfidf_vectors.indices[sorted_features]]
        return top_n_features
