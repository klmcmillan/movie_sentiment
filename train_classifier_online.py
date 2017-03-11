import sys
import numpy as np
import re
from io import open
import pyprind
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
import pickle

# initialize some preprocessing steps
stop = stopwords.words('english')

def tokenizer(text):
    """

    """
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]

    return tokenized


def stream_docs(fname):
    """

    """
    with open(fname, 'r', encoding='utf-8') as csv:
        next(csv) # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label


def get_minibatch(doc_stream, size):
    """

    """
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None

    return docs, y


def main():
    """
    User-specified parameters:
    (1) fname: name of csv file containing reviews and sentiment

    Returns:
    (1) Saves trained document classifier to file.
    """
    fname = 'movie_data.csv'

    vect = HashingVectorizer(decode_error='ignore',
                             n_features=2**21,
                             preprocessor=None,
                             tokenizer=tokenizer)
    clf = SGDClassifier(loss='log', random_state=1, n_iter=1)
    doc_stream = stream_docs(fname)

    # train in 45 batches of 1000 documents each
    pbar = pyprind.ProgBar(45)
    classes = np.array([0, 1])
    for _ in range(45):
        X_train, y_train = get_minibatch(doc_stream, size=1000)
        if not X_train:
            break
        X_train = vect.transform(X_train)
        clf.partial_fit(X_train, y_train, classes=classes)
        pbar.update()

    # classification accuracy on test set (last 5000 documents)
    X_test, y_test = get_minibatch(doc_stream, size=5000)
    X_test = vect.transform(X_test)
    print 'Accuracy: %.3f' % clf.score(X_test, y_test)

    # save classifier object
    with open('classifier_online.pkl', 'wb') as fid:
        pickle.dump(clf, fid)


if __name__ == '__main__':
    main()
    sys.exit(0)
