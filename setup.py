from setuptools import setup, find_packages

setup(
  name = 'sparselayer-tensorflow',
  packages = find_packages(),
  version = '0.0.1',
  license='MIT',
  description = 'Tensorflow 2.X implementation of Sparse Layer.',
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
