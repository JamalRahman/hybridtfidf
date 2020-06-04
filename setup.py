from distutils.core import setup
setup(
  name = 'hybridtfidf',         
  packages = ['hybridtfidf'],   
  version = '0.5',      
  license='MIT',        
  description = 'An implementation of the Hybrid TF-IDF microblog summarisation algorithm as proposed by David Ionuye and Jugal K. Kalita. Hybrid TF-IDF is designed with twitter data in mind, where document lengths are short. It is an approach to generating Multiple Post Summaries of a collection of documents.',
  author = 'Jamal Rahman',
  author_email = 'jamalrahman95@gmail.com', 
  url = 'https://github.com/jamalrahman/hybridtfidf',   
  download_url = 'https://github.com/JamalRahman/hybridtfidf/archive/v0.5.tar.gz',
  keywords = ['TFIDF', 'Text Summarization','NLP'],
  install_requires=[],
  classifiers=[
    'Development Status :: 5 - Production/Stable',    
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',  
    'Programming Language :: Python :: 3',      
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
  ],
)