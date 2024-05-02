# -*- coding: UTF-8 -*-
"""
Created on 05.11.21

:author:     Martin Dočekal
"""
import numpy
from setuptools import setup, find_packages
from Cython.Build import cythonize

with open('README.md') as readme_file:
    README = readme_file.read()

with open("requirements.txt") as f:
    REQUIREMENTS = f.read()

setup_args = dict(
    name='oapapers',
    version='1.0.0',
    description='Package for working with OAPapers dataset.',
    long_description_content_type="text/markdown",
    long_description=README,
    license='The Unlicense',
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    entry_points={
        'console_scripts': [
            'oapapers = oapapers.__main__:main'
        ]
    },
    author='Martin Dočekal',
    keywords=['dataset', 'OAPapers', 'OAPapers dataset'],
    url='https://github.com/KNOT-FIT-BUT/OAPapers',
    python_requires='>=3.9',
    install_requires=[
        "windpyutils~=2.0.12",
        "scikit-learn~=1.0.2",
        "tqdm",
        "numpy>=1.20.3",
        "spacy>=3.2.4",
        "scispacy>=0.5.1",
        "scipy>=1.7.3",
        "textual~=0.1.18",
        "textual-inputs~=0.2.6",
        "python-dateutil~=2.8.2",
        "Unidecode~=1.3.6"
    ],
    ext_modules=cythonize("oapapers/cython/*.pyx", language_level="3", language="c++"),
    include_dirs=[numpy.get_include()]
)

if __name__ == '__main__':
    setup(**setup_args)
