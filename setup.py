import os
import codecs
from setuptools import setup, find_packages


current_path = os.path.abspath(os.path.dirname(__file__))


def read_file(*parts):
    with codecs.open(os.path.join(current_path, *parts), 'r', 'utf8') as reader:
        return reader.read()



setup(
  name = 'sparselayer-tensorflow',
  packages = find_packages(),
  version = '0.0.2',
  license='MIT',
  description = 'Tensorflow 2.X implementation of Sparse Layer.',
  long_description=read_file('README.md'),
  long_description_content_type='text/markdown',
  author = 'Arya Aftab',
  author_email = 'arya.aftab@gmail.com',
  url = 'https://github.com/AryaAftab/sparselayer-tensorflow',
  keywords = [
    'deep learning',
    'tensorflow',
    'sparse layer'    
  ],
  install_requires=[
    'tensorflow>=2.2',
    'numpy>=1.19.2'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
