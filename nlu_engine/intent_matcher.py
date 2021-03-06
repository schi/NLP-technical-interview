
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

from .data_utils import DataUtils

from .tfidf_encoder import TfidfEncoder

LR = LogisticRegression(
    solver='liblinear',
    random_state=0
)
DT = DecisionTreeClassifier(random_state=42)
ADA = AdaBoostClassifier(n_estimators=100)
KN = KNeighborsClassifier(n_neighbors=100)
RF = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    random_state=0
)
SVM = svm.SVC(
    gamma='scale'
)
NB = GaussianNB()

class IntentMatcher:

    @staticmethod
    def get_dense_array(classifier, x_train):
        """
        When using NB classifier, convert the utterances to a dense array.
        :param x_train: tfidf numpy array
        :return: tfidf dense numpy array
        """

        if classifier is NB:
            print(f'{NB} has been detected, switching to a dense array.')
            x_train = x_train.todense()
        else:
            pass
        return x_train

    @staticmethod
    def train_classifier(classifier, x_train, y_train):
        print(f'Training {str(classifier)}')
        x_train = IntentMatcher.get_dense_array(classifier, x_train)
        return classifier.fit(x_train, y_train)

    @staticmethod
    def predict_labels(classifier_model, text_vectors):
        """
        Predicts the labels and returns a dataframe of the predictions
        :param classifier_model: trained classifier model
        :param tfiidf_vectorizer: trained tfidf vectorizer
        :param text_data: list of text data
        :return: predictions
        """
        print('Predicting labels')
        dense_array = IntentMatcher.get_dense_array(classifier_model, text_vectors)
        predictions = classifier_model.predict(dense_array)
        return predictions