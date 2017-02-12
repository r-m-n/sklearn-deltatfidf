=================
sklearn-deltatfidf
=================

DeltaTfidfVectorizer for scikit-learn.

The Delta TFIDF is suggested in a article_ by Justin Martineau and Tim Finin, and usually associated with sentiment classification or polarity detection of text.

Usage
-----

.. code:: python

  from sklearn_deltatfidf import DeltaTfidfVectorizer

  v = DeltaTfidfVectorizer()
  data = ['word1 word2', 'word2', 'word2 word3', 'word4']
  labels = [1, -1, -1, 1]
  v.fit_transform(data, labels)

  # you can use it in pipelines as usual
  pipe = Pipeline([
        ('vectorizer', DeltaTfidfVectorizer()),
        ('clf', svm.LinearSVC())
    ])
  pipe.fit(data, labels)

Installation
------------

With ``pip``:

.. code-block:: console

    $ pip install sklearn-deltatfidf

From source:

.. code-block:: console

    $ git clone https://github.com/r-m-n/sklearn-deltatfidf.git
    $ cd sklearn-deltatfidf
    $ python setup.py install

.. _article: http://ebiquity.umbc.edu/_file_directory_/papers/446.pdf
