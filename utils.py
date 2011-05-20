import scipy.sparse
import cPickle
import os
import sys
from model import *

operator = 'quad'
ndim = 50
nbatches = 50
lrparam = 0.1
lremb = 0.001
nbtest = 100
testall = 5
savepath = 'expe50'
simfnstr = 'dot'
listconcept = [['__produce_VB_4', '__investigation_NN_1' ,'__atlanta_NN_1' ,'__irregularity_NN_1' ,'__take_place_VB_1', '__evidence_NN_1'], ['__amateur_NN_1', '__contact_NN_1', '__collector_NN_1', '__commercial_JJ_1', '__propagandist_NN_1'], ['__american_NN_1','__knowledge_NN_1','__folklore_NN_1','__have_VB_1'],['__america_NN_1']]
listrel = [['__say_VB_1'],['__come_VB_5'],['__spread_VB_1']]
nbrank = 10
loadmodel = '/mnt/scratch/bengio/glorotxa/data/exp/glorotxa_db/wakabst2/201/model.pkl'


synset2lemme = cPickle.load(open('synset2lemme.pkl','r'))
lemme2synset = cPickle.load(open('lemme2synset.pkl','r'))
lemme2freq = cPickle.load(open('lemme2freq.pkl','r'))
synset2idx = cPickle.load(open('synset2idx.pkl','r'))
idx2synset = cPickle.load(open('idx2synset.pkl','r'))
lemme2idx = cPickle.load(open('lemme2idx.pkl','r'))
idx2lemme = cPickle.load(open('idx2lemme.pkl','r'))
synset2neg = cPickle.load(open('synset2neg.pkl','r'))
synset2def = cPickle.load(open('synset2def.pkl','r'))
synset2concept = cPickle.load(open('synset2concept.pkl','r'))
concept2synset = cPickle.load(open('concept2synset.pkl','r'))

f = open(loadmodel)
embeddings = cPickle.load(f)
leftop = cPickle.load(f)
rightop = cPickle.load(f)

simfn = eval(simfnstr+'sim')

# simi function
sl = SimilarityFunctionleftl(simfn,embeddings,leftop,rightop)
sr = SimilarityFunctionrightl(simfn,embeddings,leftop,rightop)
so = SimilarityFunctionrell(simfn,embeddings,leftop,rightop)
leftopid = Id()
rightopid = Id()
Esim = SimilarityFunctionrightl(L2sim,embeddings,leftopid,rightopid)

txt = ''
for cc in listconcept:
    txt+='\n'
    txt += getnclosest(nbrank, idx2lemme, lemme2idx, idx2synset, synset2idx, synset2def, synset2concept, concept2synset, Esim, cc, [], typ = 0, emb = True)
    for rr in listrel:
        txt+='\n'
        txt += getnclosest(nbrank,idx2lemme, lemme2idx, idx2synset, synset2idx, synset2def, synset2concept, concept2synset , sl, cc, rr, typ = 1, emb = False)
        txt+='\n'
        txt += getnclosest(nbrank,idx2lemme, lemme2idx, idx2synset, synset2idx, synset2def, synset2concept, concept2synset , sr, cc, rr, typ = 2, emb = False)
    for rr in listconcept:
        txt+='\n'
        txt += getnclosest(nbrank,idx2lemme, lemme2idx, idx2synset, synset2idx, synset2def, synset2concept, concept2synset , sr, cc, rr, typ = 3, emb = False)

print txt
