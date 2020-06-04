import math

class HybridTfidf:
    '''
        Initialise on a list of strings/documents
        fit() returns the documents as hybrid tf-idf vectors
        fit_weights() returns a list of saliency weights corresponding to each document, specifying how well
        a document represents the semantics of the documents.
        
        Both functions return the list in the same order as the initialised input list
    '''
    
    def __init__(self,docs):
        '''
        Docs: list of strings, each string is a doc
        '''
        self._post_count = len(docs)
        self._num_posts_containing_words = {}
        self._corpus_word_freqs = {}
        
        self._wordcounts(docs)
        
        self._corpus_total_word_count = sum(self._corpus_word_freqs.values())
        self._corpus_all_words = self._corpus_word_freqs.keys()
        
        self._corpus_tfidfs = self._all_tfidfs()
        
        self.raw_docs = docs
        
        
    def post_weight(self,post,threshold=6):
        '''
        This returns a post's saliency out of the collection of documents
        
        threshold: Documents normalise by length. Documents shorter than 'threshold' will normalise by the threshold.
        '''
        word_weights = []
        post = post.split(' ')
        for word in post:
            word_weight = self._corpus_tfidfs[word]
            word_weights.append(word_weight)
        return float(sum(word_weights)) / float(self._nf(post, threshold))
    
    def post_vector(self, post, threshold=6):
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
        
    def fit(self,threshold):
        '''
        threshold: Documents typically normalise by length. Documents shorter than 'threshold' will normalise by the threshold
        '''
        post_htfidf_vecs = []
        for post in self.raw_docs:
            post_htfidf_vecs.append(self.post_vector(post,threshold))
        return post_htfidf_vecs
    
    def fit_weights(self,threshold):
        '''
        threshold: Documents typically normalise by length. Documents shorter than 'threshold' will normalise by the threshold
        '''
        post_weights = []
        for post in self.raw_docs:
            post_weights.append(self.post_weight(post,threshold))
        return post_weights
        
        
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
            