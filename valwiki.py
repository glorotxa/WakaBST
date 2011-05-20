import scipy.sparse
import cPickle
import os
import sys
from model import *

synset2idx = cPickle.load(open('synset2idx.pkl','r'))
loadmodel = '/mnt/scratch/bengio/glorotxa/data/exp/glorotxa_db/wakabstfinal/98/model.pkl'

f = open(loadmodel)
embeddings = cPickle.load(f)
leftop = cPickle.load(f)
rightop = cPickle.load(f)
simfn = eval('dotsim')

#MLPout = cPickle.load(f)
#simfn = MLPout

srl = SimilarityFunctionrightl(simfn,embeddings,leftop,rightop,numpy.max(synset2idx.values())+1)
sll = SimilarityFunctionleftl(simfn,embeddings,leftop,rightop,numpy.max(synset2idx.values())+1)
sol = SimilarityFunctionrell(simfn,embeddings,leftop,rightop,numpy.max(synset2idx.values())+1)


posl = scipy.sparse.csr_matrix(cPickle.load(open('/mnt/scratch/bengio/glorotxa/data/exp/WakaBST3/Brown-synset-lhs.pkl')),dtype='float32')
posr = scipy.sparse.csr_matrix(cPickle.load(open('/mnt/scratch/bengio/glorotxa/data/exp/WakaBST3/Brown-synset-rhs.pkl')),dtype='float32')
poso = scipy.sparse.csr_matrix(cPickle.load(open('/mnt/scratch/bengio/glorotxa/data/exp/WakaBST3/Brown-synset-rel.pkl')),dtype='float32')


nbtest = 100
print calctestscore(sll,srl,sol,posl[:,:nbtest],posr[:,:nbtest],poso[:,:nbtest])


srl = SimilarityFunctionrightl(simfn,embeddings,leftop,rightop)
sll = SimilarityFunctionleftl(simfn,embeddings,leftop,rightop)
sol = SimilarityFunctionrell(simfn,embeddings,leftop,rightop)


posl = scipy.sparse.csr_matrix(cPickle.load(open('/mnt/scratch/bengio/glorotxa/data/exp/WakaBST3/Brown-synset-lhs.pkl')),dtype='float32')
posr = scipy.sparse.csr_matrix(cPickle.load(open('/mnt/scratch/bengio/glorotxa/data/exp/WakaBST3/Brown-synset-rhs.pkl')),dtype='float32')
poso = scipy.sparse.csr_matrix(cPickle.load(open('/mnt/scratch/bengio/glorotxa/data/exp/WakaBST3/Brown-synset-rel.pkl')),dtype='float32')


nbtest = 100
print calctestscore(sll,srl,sol,posl[:,:nbtest],posr[:,:nbtest],poso[:,:nbtest])



posl = scipy.sparse.csr_matrix(cPickle.load(open('/mnt/scratch/bengio/glorotxa/data/exp/WakaBST3/Brown-lemme-lhs.pkl')),dtype='float32')
posr = scipy.sparse.csr_matrix(cPickle.load(open('/mnt/scratch/bengio/glorotxa/data/exp/WakaBST3/Brown-lemme-rhs.pkl')),dtype='float32')
poso = scipy.sparse.csr_matrix(cPickle.load(open('/mnt/scratch/bengio/glorotxa/data/exp/WakaBST3/Brown-lemme-rel.pkl')),dtype='float32')


nbtest = 100
print calctestscore(sll,srl,sol,posl[:,:nbtest],posr[:,:nbtest],poso[:,:nbtest])

