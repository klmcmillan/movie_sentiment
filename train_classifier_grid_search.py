import sys
import numpy as np
import pandas as pd
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

"""
Note: from sklearn.model_selection import GridSearchCV is currently the
supported method for importing GridSearchCV, but when imported this way,
parallel processing (n_jobs=-1) doesn't work as well.
"""

# initialize some preprocessing steps
porter = PorterStemmer()
stop = stopwords.words('english')

def read_data(fname):
    """

    """
    return pd.read_csv(fname)


def preprocessor(text):
    """

    """
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')

    return text


def tokenizer(text):
    """

    """
    return text.split()


def tokenizer_porter(text):
    """

    """
    return [porter.stem(word) for word in text.split()]


def main():
    """
    User-specified parameters:
    (1) fname: name of csv file containing reviews and sentiment

    Returns:
    (1) Saves trained document classifier to file.
    """
    fname = 'movie_data.csv'

    # read and clean data
    df = read_data(fname)
    df['review'] = df['review'].apply(preprocessor)

    # divide DataFrame into training and testing sets
    X_train = df.loc[:25000, 'review'].values
    y_train = df.loc[:25000, 'sentiment'].values
    X_test = df.loc[25000:, 'review'].values
    y_test = df.loc[25000:, 'sentiment'].values

    tfidf = TfidfVectorizer(strip_accents=None,
                            lowercase=None,
                            preprocessor=None)
    param_grid = [{'vect__ngram_range': [(1, 1)],
                  'vect__stop_words': [stop, None],
                  'vect__tokenizer': [tokenizer, tokenizer_porter],
                  'clf__penalty': ['l1', 'l2'],
                  'clf__C': [1.0, 10.0, 100.0]},
                  {'vect__ngram_range': [(1, 1)],
                  'vect__stop_words': [stop, None],
                  'vect__tokenizer': [tokenizer, tokenizer_porter],
                  'vect__use_idf': [False],
                  'vect__norm': [None],
                  'clf__penalty': ['l1', 'l2'],
                  'clf__C': [1.0, 10.0, 100.0]}]
    lr_tfidf = Pipeline([('vect', tfidf),
                         ('clf', LogisticRegression(random_state=0))])
    gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                               scoring='accuracy',
                               cv=5, verbose=1,
                               n_jobs=-1)
    gs_lr_tfidf.fit(X_train, y_train)

    # print the best parameter set
    print 'Best parameter set: %s ' % gs_lr_tfidf.best_params_

    # print average cross-validation accuracy score on training set and
    # classification accuracy on test set
    print 'CV Accuracy: %.3f' % gs_lr_tfidf.best_score_
    clf = gs_lr_tfidf.best_estimator_
    print 'Test Accuracy: %.3f' % clf.score(X_test, y_test)

    # save best estimator GridSearchCV object
    with open('classifier_grid_search.pkl', 'wb') as fid:
        pickle.dump(clf, fid)


if __name__ == '__main__':
    main()
    sys.exit(0)
