from bs4 import BeautifulSoup
import pandas as pd
import os

from utils.HTML_extractor import strip_tags

class DataUtils:
    """
    This class takes care of all of the loading and basic parsing of files.
    """

    @staticmethod
    def load_html(html_file):
        """
        Loads an HTML file and returns a BeautifulSoup object.
        param: html_file: path to the HTML file
        return: BeautifulSoup object
        """

        with open(html_file, 'r') as f:
            html = f.read()
        soup = BeautifulSoup(html, 'html.parser')
        return soup

    @staticmethod
    def load_csv(csv_file):
        """
        Loads a csv file into a pandas dataframe.
        param: csv_file: path to the csv file
        return: pandas dataframe
        """
        df = pd.read_csv(csv_file)
        if {'Unnamed: 0'}.issubset(df):
            df.index = df['Unnamed: 0']
            df.drop('Unnamed: 0', axis=1, inplace=True)
            df.index.name = None
        return df

    @staticmethod
    def get_file_list(directory):
        """
        Creates a list of all of the files in a directory.
        param: directory: path to the directory
        return: list of files
        """
        file_list = []
        for file in os.listdir(directory):
            file_list.append(file)
        return file_list

    @staticmethod
    def process_html_files(directory):
        """
        Processes all of the HTML files in a directory and returns a pandas dataframe with the domain and the stripped text.
        param: directory: path to the directory
        return: pandas dataframe
        """
        df = pd.DataFrame(columns=['domain', 'text'])

        file_list = DataUtils.get_file_list(directory)

        for file in file_list:
            soup = DataUtils.load_html(directory + '/' + file)
            file_text = strip_tags(soup)
            #file_text = DataUtils.extract_text_from_html(directory + '/' + file)
            file_domain = file.replace('.html', '')
            df = df.append({'domain': file_domain, 'text': file_text}, ignore_index=True)
        return df
