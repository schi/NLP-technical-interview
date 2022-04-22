from sklearn import preprocessing
encoder = preprocessing.LabelEncoder()

class LabelEncoder:
    @staticmethod
    def encode(target_class):
        """
        Encode the labels of the dataframe.
        :param target_class: label column of a pandas dataframe
        :return: encoded labels
        """
        label_encoded_y = encoder.fit_transform(target_class.values)
        return label_encoded_y


    @staticmethod
    def inverse_transform(predicted_label):
        """
        Decode the labels of the dataframe.
        :param predicted_label: encoded labels list
        :return: decoded labels list
        """
        return encoder.inverse_transform(predicted_label)
