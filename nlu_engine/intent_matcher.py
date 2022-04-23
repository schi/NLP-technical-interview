
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

from .data_utils import DataUtils
from .entity_extractor import EntityExtractor

from .label_encoder import LabelEncoder
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
        # TODO: add in training time
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
    
    @staticmethod
    def get_incorrect_predicted_labels(data_df, classifier_model, tfidf_vectorizer):
        """
        For a data set, get the incorrect predicted labels and return a dataframe.
        """
        #TODO: implement in the analytics class
        output_df = data_df.copy()
        output_df['predicted_label'] = output_df['answer_normalised'].apply(
            lambda utterance:  IntentMatcher.predict_label(classifier_model, tfidf_vectorizer, utterance))
        return output_df[output_df['intent'] != output_df['predicted_label']]
