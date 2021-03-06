from .analytics import Analytics

import pandas as pd

class NLUEngine:
    """
    The NLUEngine class is the main class of the NLU engine, made up of intent (domain) labeling and entity extraction.
    It contains all the necessary methods to train, test and evaluate the models.
    NOTE: The entity extractor as well as other functionality has been removed for this use case.
    """

    @staticmethod
    def evaluate_intent_classifier(
        tfidf_vectors,
        labels,
        classifier,
        encoding
    ):
        """
        Evaluates a classifier and generates a report
        """

        
        predictions = Analytics.cross_validate_classifier(
            classifier,
            x_train=tfidf_vectors,
            y_train=labels
        )

        report = Analytics.generate_report(
            classifier=classifier,
            predictions=predictions,
            labels=labels
        )

        report_df = Analytics.convert_report_to_df(
            classifier=classifier,
            report=report,
            encoding=encoding
        )
        return report_df

    @staticmethod
    def evaluate_all_classifiers(classifiers, x_train, y_train, encoding):
        for count, classifier in enumerate(classifiers):
            df = NLUEngine.evaluate_intent_classifier(
                x_train, y_train, classifier, encoding)
            if count is 0:
                concat_df = df
            else:
                concat_df = pd.concat([concat_df, df])
        return concat_df
