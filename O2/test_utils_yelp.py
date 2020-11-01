from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pytest
from utils_yelp import text_clean,remove_punctuations,remove_stopwords
import nltk
nltk.download('punkt')



@pytest.mark.parametrize("x", ["D #sfsd @@"])
def test_text_clean(x):
    """Test cleaning function.
    """

    assert text_clean(x) ==  "d sfsd"


@pytest.mark.parametrize("string", ["hi."])
def test_remove_punctuations(string):

    assert remove_punctuations(string) == "hi"


@pytest.mark.parametrize("string", ["am car"])
def test_remove_stopwords(string):
    stop_words = stopwords.words('english')
    assert remove_stopwords(string,stop_words) == "car"