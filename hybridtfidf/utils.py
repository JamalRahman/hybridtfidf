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
    # print(list(zip(post_weights,enumerate(post_vectors)))[1])
    # print(sorted_keyed_vectors[0])

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
            if sim >= sim_threshold:
                is_similar = True

        if not is_similar:
            significant_indices.append(i)
            unsorted_indices.append(sorted_keyed_vectors[i][0])

        if (len(significant_indices) >= k) or (i >= veclength) :
            loop_condition = False
        i+=1
    
    return unsorted_indices


if __name__ == "__main__":
    tokenized_posts = ['suppose one way keep front line healthcare workers deaths covid bet kick think',
        'worry anyone else fix term contract',
        'great hear work hard ensure elderly people safe care home im sure greet huge relief lose family members care home think hes denial',
        'actually live free society democratic society freedom speech facebook suspend friends account delete conversation yesterday regard covid actually become communist state without realise',
        'wait package care move acute care hospital bed free massive bed capacity matter days help nhs scotland cope covid admissions',
        'governments job make profit save',
        'one go brother whilst education base sure nurse doctor also need find colour live thank',
        'coronavirus pm face labour leader uk death toll become highest europe',
        'couldnt make coronavirus top government scientist neil ferguson resign sage break lockdown rule',
        'id rather give account number sort code date birth mother maiden name someone phone tell million',
        'useful resource weve many discussions work start youtube channel frustration us traumatise pre covid also receive response covid highlight lack trauma inform care date',
        'need smart people might even say smartest people stay involve pragmatism need win',
        'love run',
        'rude order thank',
        'world coronavirus politico healthcare',
        'gb rowers plan thank regattas members row family best mitigate impact communities see info nominate amp support',
        'worthwhile hour club league would definitely recommend invest time webinar particularly advice communicate members covid crisisjust fill quote form',
        'want hear ive try keep things business usual independent business owner covid crisis tune tomorrow hear get interview team get',
        'virtue take far',
        'researchers identify potential active substances simulations drug use might also help end latest coronablog quest cure',
        'complete bollovks time study base comparisons last years death toll covid test scaremonger',
        'promote engagement learners priory youthreach',
        'deaths thoughts lose love ones',
        'professionals country uno',]

    raw_tweets = ["@ChristopherJM I suppose that's one way to keep front line healthcare workers deaths from covid-19 down. Bet @BorisJohnson is kicking himself he didn't think of this himself ",
        'Worrying for #researchears and anyone else on fixed term contracts #Covid19UK',
        'Great to hear (again) that @MattHancock has been working very hard to ensure that elderly people were safe in care homes. Im sure that will be greeted with huge relief by all who have lost family members in these same care homes - do you think hes just in denial?',
        'Do we actually live a free society, democratic society where we have freedom of speech?? If so why has Facebook suspended my friends account and deleted a conversation we had yesterday regarding Covid 19 ?? Or are we actually become a communist state without realising it',
        '@NotAMancArab Waiting on a package of care was moved out of acute care hospital beds which freed up a massive bed capacity in a matter of days. This has helped nhs Scotland cope with the covid admissions',
        "Not governments job. You made profits didn't you save any?",
        'This one goes to my brother #nhs and whilst this is education based I am sure the nurses and doctors also need to find colour in their lives - I thank you for all you are doing  #COVID19 #FE @ukfechat #EYtwittertagteam #APConnect #nurses #doctors #colour #StayHome',
        'Coronavirus: PM to face Labour leader as UK death toll becomes highest in Europe https://t.co/0s6JOFbot6 https://t.co/jTxnLZi359',
        'You couldnt make it up \n\nCoronavirus: Top government scientist Neil Ferguson resigns from SAGE after breaking lockdown rules https://t.co/AfFNEJSZN8',
        'Id rather give my account number and sort code, date of birth and mothers maiden name to someone on the phone telling me I had won $48 million. \n\n#TrackingApp #NHS #TrackBorisApp',
        '@_Sarah_Hughes_ @LizDurrant19 This is a very useful resource. Weve had many discussions at work about this. I started a YouTube channel too. There is some frustration that those of us traumatised pre-Covid should also have received this response. Covid has highlighted the lack of trauma informed care to date',
        'This. We need smart people (you might even say the SMARTEST people) to stay involved. Pragmatism needs to win out.',
        'Love running! #wednesdaywisdom #sunshine #brent #sudbury #runwithandy #fun #getdoing #staysafe #quarantine #runhappy #spring #covid19 #training #ukrunchat #may #active #trailrunning #nature #primark #healthiswealth #fitness2me #ShokzSquad #loverunning #lockdownlondon https://t.co/LpzUbWjadR',
        'Rude not to! Ordered thanks ',
        'The world after coronavirus  POLITICO https://t.co/7g9Dfnzp8I and healthcare?',
        '@boatsing The GB Rowers are planning THANK YOU Regattas for the members of the rowing family doing their best to mitigate the impact of #Covid_19 on their communities. See https://t.co/hPXrTqlr1u for info on how to nominate &amp; support. https://t.co/baa1PBxS9Z',
        'This will be a very worthwhile hour for all of our @KentFA clubs and leagues! \n\nI would definitely recommend investing the time for this webinar, particularly for advice on how to communicate with your members during this COVID-19 crisis\n\nJust fill out the quote form below ',
        'Want to hear how Ive been trying to keep things business as usual as an independent business owner during the Covid crisis?\n.\nThen tune in tomorrow at 16:00 to hear me get interviewed by the team at https://t.co/7CRctg6fKv and get https://t.co/nViiwwSfFZ',
        'Any virtue can be taken too far.',
        'Researchers from @uni_mainz_eng identify potential active substances against #coronavirus with #supercomputer simulations. Drugs used for #hepatitisC might also help https://t.co/qIyBdQX9HM More at the end of my latest coronablog, on the quest for a cure: https://t.co/1fNQJd75Mr',
        '@piersmorgan This again is complete bollovks the times study is based on comparisons to last years death tolls not Covid tests SCAREMONGER ',
        'Promoting engagement with our learners in Priory Youthreach #COVID19 #keepsafe #learningfromhome @SOLASFET @NAYCEXEC @ESF_Ireland @ddletb https://t.co/DyomLCP2ML',
        '404 #covid19 deaths. Thoughts with all who have lost loved ones. https://t.co/369y6cHpgO',
        'The professionals in our country uno']
 
    hybridtfidf = HybridTfidf(threshold=7)
    hybridtfidf.fit(tokenized_posts)
    
    post_vecs = hybridtfidf.transform(tokenized_posts)
    post_weights = hybridtfidf.transform_to_weights(tokenized_posts)

    
    most_significant = select_salient_posts(post_vecs,post_weights,3,0.15)

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
    print(tokenized_documents[0])
