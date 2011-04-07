import scipy.sparse
import cPickle
import os
import sys
from model import *

simfnstr = 'dot'
nbrank = 30
loadmodel = 'expequad1/model.pkl'


idx2concept = cPickle.load(open('idx2concept.pkl','r'))
concept2idx = cPickle.load(open('concept2idx.pkl','r'))
concept2def = cPickle.load(open('concept2def.pkl','r'))

# test set
tesl = (cPickle.load(open('WordNet3.0-test-lhs.pkl','r'))).tocsr()
tesr = (cPickle.load(open('WordNet3.0-test-rhs.pkl','r'))).tocsr()
teso = (cPickle.load(open('WordNet3.0-test-rel.pkl','r'))).tocsr()
# ------------------
rows,cols = tesl.nonzero()
idxtl = rows[numpy.argsort(cols)]
rows,cols = tesr.nonzero()
idxtr = rows[numpy.argsort(cols)]
rows,cols = teso.nonzero()
idxto = rows[numpy.argsort(cols)]

f = open(loadmodel)
embeddings = cPickle.load(f)
leftop = cPickle.load(f)
rightop = cPickle.load(f)

simfn = eval(simfnstr+'sim')


# simi function
sl = SimilarityFunctionleft(simfn,embeddings,leftop,rightop)
sr = SimilarityFunctionright(simfn,embeddings,leftop,rightop)

print calctestval(sl,sr,idxtl,idxtr,idxto)

