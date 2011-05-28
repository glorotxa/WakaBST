import scipy.sparse
import cPickle
import os
import sys
from model import *

synset2idx = cPickle.load(open('synset2idx.pkl','r'))
lemme2idx = cPickle.load(open('lemme2idx.pkl','r'))
loadmodel = '/mnt/scratch/bengio/glorotxa/data/exp/glorotxa_db/wakabstfinal4/79/model.pkl'

f = open(loadmodel)
embeddings = cPickle.load(f)
leftop = cPickle.load(f)
rightop = cPickle.load(f)
simfn = eval('dotsim')

#MLPout = cPickle.load(f)
#simfn = MLPout

srl = SimilarityFunctionrightl(simfn,embeddings,leftop,rightop,numpy.max(synset2idx.values())+1,True)
sll = SimilarityFunctionleftl(simfn,embeddings,leftop,rightop,numpy.max(synset2idx.values())+1,True)
sol = SimilarityFunctionrell(simfn,embeddings,leftop,rightop,numpy.max(synset2idx.values())+1,True)

posl = scipy.sparse.csr_matrix(cPickle.load(open('/mnt/scratch/bengio/glorotxa/data/exp/WakaBST4/XWN-lemme-lhs.pkl')),dtype='float32')
posr = scipy.sparse.csr_matrix(cPickle.load(open('/mnt/scratch/bengio/glorotxa/data/exp/WakaBST4/XWN-lemme-rhs.pkl')),dtype='float32')
poso = scipy.sparse.csr_matrix(cPickle.load(open('/mnt/scratch/bengio/glorotxa/data/exp/WakaBST4/XWN-lemme-rel.pkl')),dtype='float32')
poslc = scipy.sparse.csr_matrix(cPickle.load(open('/mnt/scratch/bengio/glorotxa/data/exp/WakaBST4/XWN-corres-lhs.pkl')),dtype='float32')
posrc = scipy.sparse.csr_matrix(cPickle.load(open('/mnt/scratch/bengio/glorotxa/data/exp/WakaBST4/XWN-corres-rhs.pkl')),dtype='float32')
posoc = scipy.sparse.csr_matrix(cPickle.load(open('/mnt/scratch/bengio/glorotxa/data/exp/WakaBST4/XWN-corres-rel.pkl')),dtype='float32')

#posl = posl[:embeddings.E.value.shape[1],:]
#posr = posr[:embeddings.E.value.shape[1],:]
#poso = poso[:embeddings.E.value.shape[1],:]

#nbtest = 100
#print calctestscore2(sll,srl,sol,posl[:,:nbtest],posr[:,:nbtest],poso[:,:nbtest])


nbtest = 100
print calctestscore3(sll,srl,sol,posl[:,:nbtest],posr[:,:nbtest],poso[:,:nbtest],poslc[:,:nbtest],posrc[:,:nbtest],posoc[:,:nbtest])

#srl = SimilarityFunctionrightl(simfn,embeddings,leftop,rightop,numpy.max(lemme2idx.values())+1,True)
#sll = SimilarityFunctionleftl(simfn,embeddings,leftop,rightop,numpy.max(lemme2idx.values())+1,True)
#sol = SimilarityFunctionrell(simfn,embeddings,leftop,rightop,numpy.max(lemme2idx.values())+1,True)

#posl = scipy.sparse.csr_matrix(cPickle.load(open('/mnt/scratch/bengio/glorotxa/data/exp/WakaBST4/XWN-synset-lhs.pkl')),dtype='float32')
#posr = scipy.sparse.csr_matrix(cPickle.load(open('/mnt/scratch/bengio/glorotxa/data/exp/WakaBST4/XWN-synset-rhs.pkl')),dtype='float32')
#poso = scipy.sparse.csr_matrix(cPickle.load(open('/mnt/scratch/bengio/glorotxa/data/exp/WakaBST4/XWN-synset-rel.pkl')),dtype='float32')

#nbtest = 100
#print calctestscore2(sll,srl,sol,posl[:,:nbtest],posr[:,:nbtest],poso[:,:nbtest])
#
#posl = scipy.sparse.csr_matrix(cPickle.load(open('/mnt/scratch/bengio/glorotxa/data/exp/WakaBST4/XWN-lemme-lhs.pkl')),dtype='float32')
#posr = scipy.sparse.csr_matrix(cPickle.load(open('/mnt/scratch/bengio/glorotxa/data/exp/WakaBST4/XWN-lemme-rhs.pkl')),dtype='float32')
#poso = scipy.sparse.csr_matrix(cPickle.load(open('/mnt/scratch/bengio/glorotxa/data/exp/WakaBST4/XWN-lemme-rel.pkl')),dtype='float32')
#
#nbtest = 100
#print calctestscore2(sll,srl,sol,posl[:,:nbtest],posr[:,:nbtest],poso[:,:nbtest])
