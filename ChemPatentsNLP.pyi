#Data used: "Golden Standard" for chemical patent mining. 249 chemical patents from www.biosemantics.org.Annotated chemical patent corpus: a gold standard for text mining.
Corpus/Article written by: Akhondi SA1, Klenner AG2, Tyrchan C3, Manchala AK4, Boppana K4, Lowe D5, Zimmermann M2, Jagarlapudi SA4, Sayle R5, Kors JA1, Muresan S6.


#LSA transform of tfidf corpus for K-Means clustering analysis with "elbow" method to determine optimal clusters. LDA Topic analysis. Indexing for latent semantic analysis plus query. Utilized gensim scikitlearn and nltk.

>>> import gensim

>>> import os

>>> import logging
>>> import nltk
>>> def iter_docs(topdir, stoplist):
    for fn in os.listdir(topdir):
        fin = open(os.path.join(topdir, fn), 'rb')
        text = fin.read()
        fin.close()
        yield (x for x in
            gensim.utils.tokenize(text, lowercase=True, deacc=True,
                                  errors="ignore")
            if x not in stoplist)
>>> class MyCorpus(object):
    def __init__(self, topdir, stoplist):
        self.topdir = topdir
        self.stoplist = stoplist
        self.dictionary = gensim.corpora.Dictionary(iter_docs(topdir, stoplist))

    def __iter__(self):
        for tokens in iter_docs(self.topdir, self.stoplist):
            yield self.dictionary.doc2bow(tokens)
>>> stoplist = set(nltk.corpus.stopwords.words("english"))
>>> corpus = MyCorpus(TEXTS_DIR, stoplist)
INFO : adding document #0 to Dictionary(0 unique tokens: [])
INFO : built Dictionary(6160 unique tokens: ['acquired', 'hydroxyisoquinoline', 'wet', 'freeze', 'states']...) from 249 documents (total 122880 corpus positions)
>>> MODELS_DIR = "/Users/danielbrickman/desktop/Patent_Corpus/models"
>>> corpus.dictionary.save(os.path.join(MODELS_DIR, "........dict"))
gensim.corpora.MmCorpus.serialize(os.path.join(MODELS_DIR, ".........mm"),
                                  corpus)
INFO : saving Dictionary object under /Users/danielbrickman/desktop/Patent_Corpus/models/mtsamples.dict, separately None
INFO : storing corpus in Matrix Market format to /Users/danielbrickman/desktop/Patent_Corpus/models/mtsamples.mm
INFO : saving sparse matrix to /Users/danielbrickman/desktop/Patent_Corpus/models/mtsamples.mm
INFO : PROGRESS: saving document #0
INFO : saved 249x6160 matrix, density=2.132% (32698/1533840)
INFO : saving MmCorpus index to /Users/danielbrickman/desktop/Patent_Corpus/models/mtsamples.mm.index
>>> tfidf = gensim.models.TfidfModel(corpus, normalize=True)
corpus_tfidf = tfidf[corpus]
INFO : collecting document frequencies
INFO : PROGRESS: processing document #0
INFO : calculating IDF weights for 249 documents and 6159 features (32698 matrix non-zeros)
>>> lsi = gensim.models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)

>>> dictionary = gensim.corpora.Dictionary.load(os.path.join(MODELS_DIR,
                                            ".......dict"))
INFO : loading Dictionary object from /Users/danielbrickman/desktop/Patent_Corpus/models/mtsamples.dict
>>> corpus = gensim.corpora.MmCorpus(os.path.join(MODELS_DIR, "......mm"))
INFO : loaded corpus index from /Users/danielbrickman/desktop/Patent_Corpus/models/mtsamples.mm.index
INFO : initializing corpus reader from /Users/danielbrickman/desktop/Patent_Corpus/models/mtsamples.mm
INFO : accepted corpus with 249 documents, 6160 features, 32698 non-zero entries
>>> tfidf = gensim.models.TfidfModel(corpus, normalize=True)
INFO : collecting document frequencies
INFO : PROGRESS: processing document #0
INFO : calculating IDF weights for 249 documents and 6159 features (32698 matrix non-zeros)
>>> corpus_tfidf = tfidf[corpus]
>>> lsi = gensim.models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)
INFO : using serial LSI version on this node
INFO : updating model with new documents
INFO : preparing a new chunk of documents
INFO : using 100 extra samples and 2 power iterations
INFO : 1st phase: constructing (6160, 102) action matrix
INFO : orthonormalizing (6160, 102) action matrix
INFO : 2nd phase: running dense svd on (102, 249) matrix
INFO : computing the final decomposition
INFO : keeping 2 factors (discarding 71.361% of energy spectrum)
INFO : processed documents up to #249
INFO : topic #0(6.660): 0.423*"hz" + 0.341*"j" + 0.296*"h" + 0.259*"singlet" + 0.203*"triplet" + 0.177*"multiplet" + 0.174*"doublet" + 0.157*"phenyl" + 0.154*"methylethyl" + 0.140*"yl"
INFO : topic #1(4.558): 0.435*"tz" + 0.405*"bu" + 0.373*"et" + 0.365*"pr" + 0.344*"h" + 0.298*"cooh" + 0.178*"hooc" + 0.124*"tbu" + 0.100*"ipr" + 0.090*"mes"
>>> fcoords = open(os.path.join(MODELS_DIR, "coords.csv"), 'wb')
for vector in lsi[corpus]:
    if len(vector) != 2:
        continue
    fcoords.write(b"%6.4f\t%6.4f\n" % (vector[0][1], vector[1][1]))
fcoords.close()
>>> import matplotlib.pyplot as plt
>>> from sklearn.cluster import KMeans
>>> MAX_K = 10
>>> X = np.loadtxt(os.path.join(MODELS_DIR, "coords.csv"), delimiter="\t")
>>> ks = range(1, MAX_K + 1)
>>> inertias = np.zeros(MAX_K)
>>> diff = np.zeros(MAX_K)
>>> diff2 = np.zeros(MAX_K)
>>> diff3 = np.zeros(MAX_K)
>>> for k in ks:
    kmeans = KMeans(k).fit(X) #good explanation of inertial here: http://scikit-learn.org/stable/modules/clustering.html
    #Vincent Granville's approach of calculating the third differential to find an elbow point. http://www.analyticbridge.com/profiles/blogs/identifying-the-number-of-clusters-finally-a-solution
    #First read about here: http://sujitpal.blogspot.com/2014/08/topic-modeling-with-gensim-over-past.html
    inertias[k - 1] = kmeans.inertia_
    # first difference
    if k > 1:
        diff[k - 1] = inertias[k - 1] - inertias[k - 2]
    # second difference
    if k > 2:
        diff2[k - 1] = diff[k - 1] - diff[k - 2]
    # third difference
    if k > 3:
        diff3[k - 1] = diff2[k - 1] - diff2[k - 2]

elbow = np.argmin(diff3[3:]) + 3

plt.plot(ks, inertias, "b*-")
plt.plot(ks[elbow], inertias[elbow], marker='o', markersize=12,
         markeredgewidth=2, markeredgecolor='r', markerfacecolor=None)
plt.ylabel("Inertia")
plt.xlabel("K")
plt.show()
>>> NUM_TOPICS = 5

X = np.loadtxt(os.path.join(MODELS_DIR, "coords.csv"), delimiter="\t")
kmeans = KMeans(NUM_TOPICS).fit(X)
y = kmeans.labels_

colors = ["b", "g", "r", "m", "c"]
for i in range(X.shape[0]):
    plt.scatter(X[i][0], X[i][1], c=colors[y[i]], s=10)
plt.show()

>>> lda = gensim.models.LdaModel(corpus, id2word=dictionary, num_topics=NUM_TOPICS)
lda.print_topics(NUM_TOPICS)
INFO : using symmetric alpha at 0.2
INFO : using symmetric eta at 0.2
INFO : using serial LDA version on this node
INFO : running online LDA training, 5 topics, 1 passes over the supplied corpus of 249 documents, updating model once every 249 documents, evaluating perplexity every 249 documents, iterating 50x with a convergence threshold of 0.001000
WARNING : too few updates, training might not converge; consider increasing the number of passes or iterations to improve accuracy
INFO : -9.692 per-word bound, 827.2 perplexity estimate based on a held-out corpus of 249 documents with 122880 words
INFO : PROGRESS: pass 0, at document #249/249
INFO : topic #0 (0.200): 0.046*h + 0.012*j + 0.012*et + 0.011*r + 0.011*hz + 0.011*g + 0.011*methyl + 0.010*group + 0.010*n + 0.010*reaction
INFO : topic #1 (0.200): 0.030*h + 0.024*c + 0.017*j + 0.015*hz + 0.015*n + 0.014*r + 0.014*compound + 0.014*yl + 0.012*phenyl + 0.011*g
INFO : topic #2 (0.200): 0.136*h + 0.019*et + 0.017*bu + 0.015*pr + 0.015*tz + 0.014*j + 0.013*hz + 0.013*phenyl + 0.012*singlet + 0.010*yl
INFO : topic #3 (0.200): 0.019*group + 0.018*c + 0.017*r + 0.015*reaction + 0.013*h + 0.011*groups + 0.010*ml + 0.010*alkyl + 0.009*carbon + 0.009*compound
INFO : topic #4 (0.200): 0.043*h + 0.018*c + 0.014*g + 0.014*methyl + 0.014*str + 0.013*reaction + 0.013*hz + 0.011*yl + 0.011*compound + 0.010*j
INFO : topic diff=2.530300, rho=1.000000
INFO : topic #0 (0.200): 0.046*h + 0.012*j + 0.012*et + 0.011*r + 0.011*hz + 0.011*g + 0.011*methyl + 0.010*group + 0.010*n + 0.010*reaction
INFO : topic #1 (0.200): 0.030*h + 0.024*c + 0.017*j + 0.015*hz + 0.015*n + 0.014*r + 0.014*compound + 0.014*yl + 0.012*phenyl + 0.011*g
INFO : topic #2 (0.200): 0.136*h + 0.019*et + 0.017*bu + 0.015*pr + 0.015*tz + 0.014*j + 0.013*hz + 0.013*phenyl + 0.012*singlet + 0.010*yl
INFO : topic #3 (0.200): 0.019*group + 0.018*c + 0.017*r + 0.015*reaction + 0.013*h + 0.011*groups + 0.010*ml + 0.010*alkyl + 0.009*carbon + 0.009*compound
INFO : topic #4 (0.200): 0.043*h + 0.018*c + 0.014*g + 0.014*methyl + 0.014*str + 0.013*reaction + 0.013*hz + 0.011*yl + 0.011*compound + 0.010*j

##AND NOW for LSA INDEXING (Latent Semantic Analysis)

index = similarities.SparseMatrixSimilarity(corpus_tfidf, num_features = 30)

>>> from gensim.similarities import MatrixSimilarity, SparseMatrixSimilarity, Similarity
>>> corpus_lsi = lsi[corpus_tfidf]

>>> import scipy.sparse
>>> import numpy
>>> index_dense = MatrixSimilarity(corpus_lsi, num_features=lsi.num_terms)
INFO : creating matrix with 249 documents and 6160 features

#SAVE. LOAD.

>>> index.save('/Users/danielbrickman/Desktop/index_dense.index')
INFO : saving SparseMatrixSimilarity object under /Users/danielbrickman/Desktop/index_dense.index, separately None
>>> indexl = Similarity.load('/Users/danielbrickman/Desktop/index_dense.index')
INFO : loading Similarity object from /Users/danielbrickman/Desktop/index_dense.index

#Cosine Similarity for Search Query. (Document Number, %Similar for top 5 Documents)

>>> query = "organic extracted dried over sodium sulfate concentrated vacuo"

>>> query_bow = dictionary.doc2bow(gensim.utils.tokenize(query))
>>> query_tfidf = tfidf[query_bow]
>>> query_lsi = lsi[query_tfidf]
>>> index_dense.num_best = 5
>>> print(index_dense[query_lsi])
[(213, 0.99999988079071045), (219, 0.99999982118606567), (198, 0.99999028444290161), (207, 0.99998247623443604), (199, 0.99997228384017944)]
>>> query_bow = corpus.doc2bow(tokenize(query))



