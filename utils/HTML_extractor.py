import re
from io import StringIO

from html.parser import HTMLParser
from bs4 import BeautifulSoup

class HTMLExtractor(HTMLParser):
    """
    A class that easily turns HTML into text, without tags or other attributes (like line breaks).
    """
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.text = StringIO()
    def handle_data(self, d):
        self.text.write(d)
    def get_data(self):
        return self.text.getvalue()


def strip_tags(soup_object):
    """
    Gets text from a BeautifulSoup object, strips all tags, and replaces any length of line breaks with one space.
    """
    extractor = HTMLExtractor()
    html = soup_object.get_text()
    extractor.feed(html)
    text = extractor.get_data()
    text = re.sub(r'[\n\t\xa0\r\s]+', ' ', text)
    return text
