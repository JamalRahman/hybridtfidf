from numpy.linalg import norm
from numpy import dot

def cosine_sim(vec1, vec2):
    '''Calculates the cosine similarity between two vectors
    
    Args:
        vec1 (list of float): A vector
        vec2 (list of float): A vector

    Returns:
        The cosine similarity between the two input vectors 
    '''
    return dot(vec1,vec2) / (norm(vec1)*norm(vec2))


def select_salient_posts(post_vectors,post_weights,k = 10,similarity_threshold = 0.4):
    '''
        Selects the top k most salient posts in a collection of posts.
        To avoid redundancy, any post too similar to other-posts are disregarded. Each selected post will therefore be both highly salient and representative of unique semantics.

        Note:
            post_vectors and post_weights must be in the same order. The ith element of post_weights must reflect the ith element of post_vectors

        Args:
            post_vectors (list of (list of float)): Hybrid tfidf representation of the documents as a document-term matrix

            post_weights (list of float): Hybrid Tfidf weight for each document

            k (int): The number of posts to select as output

            sim_threshold (float): The maximum cosine similiarity for a post to be selected

        Returns:

    '''


    sorted_keyed_vectors = [z for _, z in sorted(zip(post_weights,enumerate(post_vectors)), key=lambda i: i[0],reverse=True)] # z is (i,vi) sorted by weight

    i = 1

    veclength = len(post_vectors)
    loop_condition = True

    significant_indices = []
    significant_indices.append(0)
    unsorted_indices = []
    unsorted_indices.append(sorted_keyed_vectors[0][0])

    while loop_condition:
        is_similar = False

        for j in significant_indices:
            sim = cosine_sim(sorted_keyed_vectors[j][1], sorted_keyed_vectors[i][1])
            if sim >= similarity_threshold:
                is_similar = True

        if not is_similar:
            significant_indices.append(i)
            unsorted_indices.append(sorted_keyed_vectors[i][0])

        if (len(significant_indices) >= k) or (i >= veclength-1) :
            loop_condition = False
        i+=1
    
    return unsorted_indices