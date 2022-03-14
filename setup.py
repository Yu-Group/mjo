# build for release: python -m build
# build for development: python setup.py develop

import setuptools
from os import path

def read(fname):
    return open(path.join(path.dirname(__file__), fname)).read()

setuptools.setup(
    name='mjonet',
    version='0.1',
    description='Interpretable forecasting of the Madden-Julian Oscillation',
    long_description=read('README.md'),
    long_description_content_type="text/markdown",
    author='James Duncan',
    author_email='jpduncan@berkeley.edu',
    url='https://github.com/Yu-Group/mjonet',
    packages=setuptools.find_packages(exclude=['tests']),
    install_requires=[
        'torch>=1.6',
        'numpy',
    ],
    python_requires='>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
