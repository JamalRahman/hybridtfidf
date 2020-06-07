from setuptools import setup, find_packages


from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
  name = 'hybridtfidf',
  packages = find_packages(),
  version = '1.0.2',
  license='MIT',
  description = 'An implementation of the Hybrid TF-IDF microblog summarisation algorithm as proposed by David Ionuye and Jugal K. Kalita.',
  long_description = long_description,
  long_description_content_type = 'text/markdown',
  author = 'Jamal Rahman',
  author_email = 'jamalrahman95@gmail.com', 
  url = 'https://github.com/jamalrahman/hybridtfidf',
  keywords = ['TFIDF', 'Text Summarization','NLP'],
  classifiers=[
    'Development Status :: 5 - Production/Stable',    
    'Intended Audience :: Developers',
    'License :: OSI Approved :: MIT License',  
    'Programming Language :: Python :: 3',
  ],
)