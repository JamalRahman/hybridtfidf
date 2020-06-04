import math

class HybridTfidf:
    '''
        Initialise on a list of strings/documents
        fit() returns the documents as hybrid tf-idf vectors
        fit_weights() returns a list of saliency weights corresponding to each document, specifying how well
        a document represents the semantics of the documents.
        
        Both functions return the list in the same order as the initialised input list
    '''
    
    def __init__(self,threshold = 5):
        self._isfit = False
        self._threshold = threshold
        
        
    def fit_transform(self, raw_documents):
        '''Learn vocabulary and hybrid tf-idf terms, return document-term matrix

        This is equivalent to calling fit and then transform

        Args:
            raw_documents (list of str): List of documents

        Returns:
            List of documents transformed into their Hybrid TF-IDF vector representations
        '''
        self.fit(raw_documents)
        self.transform(self._raw_documents)

    def fit(self, raw_documents):
        '''Learn vocabulary and calculate hybrid tf-idf terms for the vocabulary

        Args:
            raw_documents (list of str): List of documents

        '''

        self._post_count = len(raw_documents)
        self._num_posts_containing_words = {}
        self._corpus_word_freqs = {}
        self._raw_documents = raw_documents

        self._wordcounts(raw_documents)
        
        self._corpus_total_word_count = sum(self._corpus_word_freqs.values())
        self._corpus_all_words = self._corpus_word_freqs.keys()
        
        self._corpus_tfidfs = self._all_tfidfs()
        self._isfit = True


    def transform(self, raw_documents):
        '''Transform documents to a document-term matrix

        Uses the vocabulary and term/document frequencies learned by fit or fit_transform.

        Args:
            raw_documents (list of str): List of documents.
        
        Returns:
            List of documents transformed into their Hybrid TF-IDF vector representations

        '''
        post_htfidf_vecs = []
        for post in raw_documents:
            post_htfidf_vecs.append(self.post_vector(post,self._threshold))
        return post_htfidf_vecs


    def transform_to_weights(self, post_vectors):
        '''Construct the post-saliency vector by aggregating normalised term-saliencies

        Args:
            post_vectors (list of (list of float)): Hybrid tfidf representation of the documents as a document-term matrix

        Returns:
            List of weights (floats) that represent the saliency of each document in the fit dataset. 

        '''
        post_weights = [sum(vec) for vec in post_vectors]
        return post_weights
    


    # ---- Vector construction -------------------

    def post_weight(self,post,threshold=self._threshold):
        '''
        This returns a post's saliency out of the collection of documents
        
        threshold: Documents normalise by length. Documents shorter than 'threshold' will normalise by the threshold.
        '''
        word_weights = []
        post = post.split(' ')
        for word in post:
            word_weight = self._corpus_tfidfs[word]
            word_weights.append(word_weight)
        return float(sum(word_weights)) / float(self._nf(post, self._threshold))
    
    def post_vector(self, post, threshold=self._threshold):
        '''
        This returns the tfidf word vector for a post.
        Output elements should be in the order corresponding to the word-order of self._corpus_tfids.keys()==self._corpus_all_words
        '''
        vec = []
        normalisation_factor = float(self._nf(post,threshold))
        splitpost = post.split(' ')
        
        for word,tfidf in self._corpus_tfidfs.items():
            
            normalised_tfidf = float(tfidf) / normalisation_factor
            normalised_tfidf = normalised_tfidf * (word in splitpost)

            vec.append(normalised_tfidf)
        return vec

    def get_feature_names(self):
        ''' Outputs the vocabulary in the order that all tfidf-vectors use

        Returns:
            List of the vocabulary which the object was trained on
        '''

        # TODO: ERROR hANDLE FOR ENSURING FIT


        return self._corpus_all_words

    # ---- TFIDF ----------------
    
    def tfidf_word_weight(self,word):
        '''
        Under Hybrid TFIDF, words have (mostly) the same TFIDF regardless of what document they're in.
        (Not calculated here but tfidf word weights can be normalised by length if returning post_vector)
        '''
        return self._tf(word)*math.log(self._idf(word),2)
    
    def _tf(self,word):
        return float(self._corpus_word_freqs[word]) / float(self._corpus_total_word_count)
           
    def _idf(self,word):
        return float(self._post_count) / float(self._num_posts_containing_words[word])
           
    def _nf(self,post,threshold):
        return max(threshold,len(post))
    
    # ---- MEMOIZATION ---------
    
    def _all_tfidfs(self):
        
        all_tfidfs = {}
        
        for word in self._corpus_all_words:
            word_tfidf = self.tfidf_word_weight(word)
            all_tfidfs[word] = word_tfidf
        
        return all_tfidfs

    
    def _wordcounts(self, docs):
        for doc in docs:
            seen_words = set()
            for word in doc.split(' '):
                try:
                    self._corpus_word_freqs[word]+=1
                except:
                    self._corpus_word_freqs[word]=1
                seen_words.add(word)
            for word in seen_words:
                try:
                    self._num_posts_containing_words[word]+=1
                except:
                    self._num_posts_containing_words[word]=1
            