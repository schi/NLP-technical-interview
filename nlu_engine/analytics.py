import time
import pandas as pd

from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from sklearn_crfsuite.metrics import flat_classification_report

from .data_utils import DataUtils

class Analytics:
    @staticmethod
    def cross_validate_classifier(classifier, x_train, y_train):
        start = time.time()
        print(f'Cross validating with {str(classifier)}')

        x_train = DataUtils.get_dense_array(classifier, x_train)
        
        prediction = cross_val_predict(
            estimator=classifier,
            X=x_train,
            y=y_train,
            cv=5
        )
        stop = time.time()
        duration = stop - start
        print(f'Time it took to cross validate {str(classifier)}: {duration}')
        return prediction

    @staticmethod
    def generate_report(classifier, predictions, labels):
        # TODO: change function to staticmethod

        report = classification_report(
            y_pred=predictions,
            y_true=labels,
            output_dict=True
        )
        print(f'Generating report for {classifier}')
        return report

    @staticmethod
    def convert_report_to_df(classifier, report, encoding):
        """
        converts reports to a dataframe and adds the label of the prediction (i.e. intents or domains), encoding of the utterances (i.e. tfidf or word2vec), and the classifier used.
        """
        df = pd.DataFrame(report).transpose()
        df['classifier'] = str(classifier)
        df['classifier'] = df['classifier'].str.replace(r"\([^()]*\)", "")

        df.index = df.index.set_names(['is_shop'])
        
        df = df.reset_index()
        output_df = df.copy()
        output_df['encoding'] = encoding
        return output_df
