#!/usr/bin/env python
# coding: utf-8

# Author: Ge Zhou
# Email: ge.zhou@kaust.edu.sa

from setuptools import setup, find_packages

setup(
    name='my_ml_project',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'scipy',
        'pandas',
        'numpy',
        'xgboost',
        'scikit-learn',
        'shap',
        'matplotlib',
    ],
)
