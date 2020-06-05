# Hybrid TF-IDF
---

This is an implementation of the [Hybrid TF-IDF algorithm](https://ieeexplore.ieee.org/document/6113128) as proposed by David Ionuye and Jugal K. Kalita (2011).

Hybrid TF-IDF is designed with twitter data in mind, where document lengths are short. It is an approach to generating Multiple Post Summaries of a collection of documents.

Simply install with:
```
pip install hybridtfidf
```
Load some short texts of the form:
```
documents = ['This is one example of a short text.',
            'Designed for twitter posts, a typical 'short document' will have fewer than 280 characters!'
            ]
```
---
The algorithm works best on tokenized data with stopwords removed, although this is not required. You can tokenize your documents any way you like. Here is an example using the popular [NLTK](https://www.nltk.org/) package:

```
import nltk
nltk.download('stopwords')

documents = ["This is one example of a short text.",
            "Designed for twitter posts, a typical 'short document' will have fewer than 280 characters!"
            ]

stop_words = set(nltk.corpus.stopwords.words('english'))

tokenized_documents = []

for document in documents:
    tokens = nltk.tokenize.word_tokenize(document)
    tokenized_document = [i for i in tokens if not i in stop_words]
    tokenized_documents.append(tokenized_document)    

# tokenized_documents[0] = ['This','one','example','short','text','.']
```

The algorithm however requires that each document is one string. If you use nltk's tokenizer, make sure to re-join each document string.

```
tokenized_documents = [' '.join(document) for document in tokenized_documents]

# tokenized_documents[0] = 'This one example short text .'
```

---

Create a HybridTfidf object and fit it on the data

```
hybridtfidf = HybridTfidf(threshold=7)
hybridtfidf.fit(tokenized_documents)

# The thresold value affects how strongly the algorithm biases towards longer documents
# A higher threshold will make longer documents have a higher post weight
# (see next snippits of code for what post weight does)
```

Transform the documents into their Hybrid TF-IDF vector representations, and get the saliency values for each document.
```
post_vectors = hybridtfidf.transform(tokenized_posts)
post_weights = hybridtfidf.transform_to_weights(tokenized_posts)
```
The post vectors represent the documents as embedded in Hybrid TF-IDF vector space, any linear algebra techniques can be performed on these!

The post weights list gives you a single number for each document, this number reflects how *salient* each document is (how strongly the document contributes towards a topical discussion). In theory, spammy-documents will have a low post saliency weight. 

Lastly, Ionuye and Kalita proposed using Hybrid TF-IDF to summarise the collection of documents.
We select 'k' of the most relevant/salient documents, and to avoid redundancy we do not select any documents which are too cosine-similar to previous documents. In effect we select the top 'k' most important documents, skipping over documents that talk about the same topic. I.e - we summarise the collection of documents into 'k' representative documents.

```
# Get the indices of the most significant documents. 
most_significant = select_salient_posts(post_vectors,post_weights, k = 5, similarity_threshold = 0.5)

for i in most significant:
    print(documents[i])         # Prints the 'k' most significant documents that are each about a separate topic
```


Note: The indices of: the fit() input (the starting document list), the post_vectors, and the post_weights, are all lined up. Make sure not to re-order one without re-ordering the others similarly.