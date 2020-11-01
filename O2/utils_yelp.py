# Read data
from string import punctuation
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from nltk.stem.porter import PorterStemmer
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
def loadJSON(path, data_dict,size=-1):
    """
        Function that loads data. Since the reviews is not JSON itself, but each row it is, i need to read it line by line.
        :param path: path to file
        :param data_dict: columns:list
        :param size: how many lines to load
        :rtype size: int
        :rtype path: str
        :rtype data_dict: dict
        :return: data with values read.

    """

    # check input
    if len(data_dict) == 0:
        raise RuntimeError("No columns")

    else:
        cnt = -1
        # data are to big to load in normal way, seoncondly, it seems no correct json format.
        with open(path, 'rb') as f:
            from json import loads
            if size !=-1:
                cnt = 0
            for line in f:
                cnt+=1
                line = loads(line)
                # not happy about nested loops, but for now it will do.
                for key in data_dict.keys():
                    # not using get method, since i need to raise error if key not exists
                    data_dict[key].append(line[key])
                del line
                if cnt > 0 and cnt == size  :
                    break
            # with contectx should do work but just in case
            f.close()
    return data_dict


def text_clean(review):
    """Process review into token
       remove following regex

    # remove hypertext links
    review = re.sub(r'https?:\/\/.*[\r\n]*', '', review)
    # extract hash tag
    review = re.sub(r'@', '', review)
    # extract @
    review = re.sub(r'#', '', review)
    # extract numbers
    review = re.sub('[0-9]*[+-:]*[0-9]+', '', review)
    # extract '
    review = re.sub("'s", "", review)

    strip empty spaces and lower case words.

       :param review: the review.
       :rtype review: string

       :return list_of_words: list with words cleaned fro mthe review.
    """

    import re

    # remove hypertext links
    review = re.sub(r'https?:\/\/.*[\r\n]*', '', review)
    # extract hash tag
    review = re.sub(r'@', '', review)
    # extract @
    review = re.sub(r'#', '', review)
    # extract numbers
    review = re.sub('[0-9]*[+-:]*[0-9]+', '', review)
    # extract '
    review = re.sub("'s", "", review)
    return review.strip().lower()


def remove_punctuations(string):
    """
    Remove puctuation.
    :param string:
    :return: string
    """
    return ''.join(c for c in string if c not in punctuation)



def remove_stopwords(string, stop_words):
    """
    Removing stop words.
    :param string:
    :param stop_words:
    :return: string
    """
    tokenized = word_tokenize(string)
    filtered_sentence = [word for word in tokenized if not word in stop_words]
    return ' '.join(c for c in filtered_sentence)


def sigmoid(z):
    '''
    Input:
        z: is the input (can be a scalar or an array)
    Output:
        h: the sigmoid of z
    '''

    # calculate the sigmoid of z
    h = 1 / (1 + np.exp(-z))
    return h

def print_significant_words(logreg_coef=None, class_=None, count=100, count_vector=None, graph=True):
    """

    :param logreg_coef: logistic regression coefficients
    :param class_:  0 is negative, 1 positive
    :param count: how many words to show, default to 100
    :param count_vector: counting vector
    :param graph: True ro print graph
    :rtype logreg_coef: numpy array
    :return:
    """


    if isinstance(logreg_coef, np.ndarray) and class_ in (0, 1) and isinstance(count_vector, CountVectorizer):
        pass
    else:
        raise TypeError("Parameters has wrong type, see help(print_significant_words)")

    # get id from model
    if class_ == 0:
        # since we sort ids, we choose range below or above 0
        # for negative sentiment, estimator should have + sing
        range_ = range(0, 1 + count, 1)
        sentiment = 'Negative'
    else:

        # since we sort ids, we choose range below or above 0
        # for positive sentiment, estimator should have + sing
        range_ = range(-1, -1 - count, -1)
        sentiment = 'Positive'
    ids = np.argsort(logreg_coef)

    words = [list(count_vector.vocabulary_.keys())[list(count_vector.vocabulary_.values()).index(id_)] for id_ in
             ids[range_]]

    # graph
    if graph == True:
        fig = plt.figure(figsize=(10, 6))
        ax = sns.barplot(words, logreg_coef[ids[range_]])
        plt.title("Top {} words for {} Sentiment".format(count, sentiment), fontsize=20)
        x_locs, x_labels = plt.xticks()
        plt.setp(x_labels, rotation=40)
        plt.ylabel('Feature Importance', fontsize=12)
        plt.xlabel('Word', fontsize=12);

    return words

def wc(words):
    """

    :param words: list of word to make fancy graph
    :type words: list
    :return:
    """

    wordcloud = WordCloud(background_color="white", max_words=len(' '.join(words)), \
                          max_font_size=40, relative_scaling=.5, colormap='summer').generate(' '.join(words))
    plt.figure(figsize=(13, 13))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()


def stemming(str_input):
    """
    Stemmer. Removing ing , ed etc....
    :param str_input: Input sentence
    :return: string of stemmed words
    """

    words= word_tokenize(str_input)
    porter_stemmer = PorterStemmer()
    filtered_sentence = [porter_stemmer.stem(word) for word in words]
    return ' '.join(c for c in filtered_sentence)


if __name__ == '__main__':
    pass