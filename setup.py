# -*- coding: utf-8 -*-
from setuptools import setup

setup(
    name='sklearn-deltatfidf',
    version=0.1,
    packages=['sklearn_deltatfidf'],
    description='DeltaTfidfVectorizer for scikit-learn',
    url='https://github.com/r-m-n/sklearn-deltatfidf',
    download_url='https://github.com/r-m-n/sklearn-deltatfidf/archive/0.1.tar.gz',
    keywords=['sklearn', 'scikit-learn', 'tfidf', 'deltatfidf', 'delta tfidf'],
    license='MIT',
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
        'six'
    ]
)
