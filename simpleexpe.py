import scipy.sparse
import cPickle
import os
import sys
from model import *

train = 'train'
operator = 'quad'
ndim = 100
nbatches = 10
lrparam = 1
lremb = 0.01
nbtest = 100 
testall = 25
savepath = 'expe100'
simfnstr = 'dot'
listconcept = ['__brain_NN_1', '__eat_VB_1', '__france_NN_1', '__auto_NN_1']
listrel = ['_has_part','_similar_to','_member_of_domain_topic','_part_of','_verb_group','_derivationally_related_form']
nbrank = 30
loadmodel = False#'expe100/model.pkl'


print >> sys.stderr, 'train set : ', train
print >> sys.stderr, 'operator : ', operator
print >> sys.stderr, 'ndim : ',  ndim
print >> sys.stderr, 'nbbatches : ', nbatches
print >> sys.stderr, 'lrparam : ', lrparam
print >> sys.stderr, 'lremb : ', lremb
print >> sys.stderr, 'nbtest : ', nbtest
print >> sys.stderr, 'testall : ', testall
print >> sys.stderr, 'savepath : ', savepath
print >> sys.stderr, 'simfnstr : ', simfnstr
print >> sys.stderr, 'listconcept : ', listconcept
print >> sys.stderr, 'listrel : ', listrel
print >> sys.stderr, 'nbrank : ', nbrank
print >> sys.stderr, 'loadmodel : ', loadmodel


if savepath not in os.listdir('.'):
    os.mkdir(savepath)

idx2concept = cPickle.load(open('idx2concept.pkl','r'))
concept2idx = cPickle.load(open('concept2idx.pkl','r'))
concept2def = cPickle.load(open('concept2def.pkl','r'))
amb2idx = cPickle.load(open('amb2idx.pkl','r'))
idx2amb = cPickle.load(open('idx2amb.pkl','r'))

# train set
posl = (cPickle.load(open('WordNet3.0-%s-lhs.pkl'%train,'r'))).tocsr()
posr = (cPickle.load(open('WordNet3.0-%s-rhs.pkl'%train,'r'))).tocsr()
poso = (cPickle.load(open('WordNet3.0-%s-rel.pkl'%train,'r'))).tocsr()

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

# random
random = scipy.sparse.lil_matrix((len(idx2concept.keys()+idx2amb.keys()),2*posl.shape[1]),dtype=theano.config.floatX)
idxr = numpy.asarray(numpy.random.randint(len(idx2concept.keys()+idx2amb.keys()),size=(2*posl.shape[1])),dtype='int32')
for idx,i in enumerate(idxr):
    random[i,idx]=1

print posl.shape,posr.shape,poso.shape,random.shape


random = random.tocsr()


if not loadmodel:
    # operators
    if operator == 'Id':
        leftop = Id()
        rightop = Id()
    elif operator == 'linear':
        leftop = Layercomb(numpy.random, 'lin', ndim, ndim, ndim)
        rightop = Layercomb(numpy.random, 'lin', ndim, ndim, ndim)
    elif operator == 'mlp':
        leftop = MLP(numpy.random, 'rect', ndim, ndim, (3*ndim)/2, ndim)
        rightop = MLP(numpy.random, 'rect', ndim, ndim, (3*ndim)/2, ndim)
    elif operator == 'quad':
        leftop = Quadlayer(numpy.random, ndim, ndim, (3*ndim)/2, ndim)
        rightop = Quadlayer(numpy.random, ndim, ndim, (3*ndim)/2, ndim)
    # embeddings
    embeddings = Embedd(numpy.random,len(idx2concept.keys()+idx2amb.keys()),ndim)
else:
    f = open(loadmodel)
    embeddings = cPickle.load(f)
    leftop = cPickle.load(f)
    rightop = cPickle.load(f)

simfn = eval(simfnstr+'sim')


# train function
ft = TrainFunction(simfn,embeddings,leftop,rightop)

# simi function
sl = SimilarityFunctionleft(simfn,embeddings,leftop,rightop)
sr = SimilarityFunctionright(simfn,embeddings,leftop,rightop)
leftopid = Id()
rightopid = Id()
Esim = SimilarityFunctionright(L2sim,embeddings,leftopid,rightopid)

ct = 0
M = posl.shape[1]/nbatches

left = []
right = []
leftb = []
rightb = []

while 1:
    for i in range(nbatches):
        resl = ft(lrparam/float(M),lremb,posl[:,i*M:(i+1)*M],posr[:,i*M:(i+1)*M],poso[:,i*M:(i+1)*M],random[:,2*i*M:(2*i+1)*M],random[:,(2*i+1)*M:(2*i+2)*M])
        left += [resl[1]/float(M)]
        right += [resl[2]/float(M)] 
        leftb += [resl[4]/float(M)]
        rightb += [resl[5]/float(M)]
        embeddings.norma()
    order = numpy.random.permutation(posl.shape[1])
    posl,posr,poso = (posl[:,order],posr[:,order],poso[:,order])
    random = random[:,numpy.random.permutation(random.shape[1])]
    ct = ct + 1
    if ct/float(testall) == ct / testall:
        print >> sys.stderr, '------ Epoch ', ct
        print >> sys.stderr, numpy.mean(left+right), numpy.std(left+right),numpy.mean(left),numpy.std(left),numpy.mean(right), numpy.std(right)
        print >> sys.stderr, numpy.mean(leftb+rightb), numpy.std(leftb+rightb),numpy.mean(leftb),numpy.std(leftb),numpy.mean(rightb), numpy.std(rightb)
        txt = ''
        txt += '%s %s %s %s %s %s\n'%(numpy.mean(left+right), numpy.std(left+right),numpy.mean(left),numpy.std(left),numpy.mean(right), numpy.std(right))
        txt += '%s %s %s %s %s %s\n'%(numpy.mean(leftb+rightb), numpy.std(leftb+rightb),numpy.mean(leftb),numpy.std(leftb),numpy.mean(rightb), numpy.std(rightb))
        left = []
        right = []
        leftb = []
        rightb = []
        result = calctestval(sl,sr,idxtl[:nbtest],idxtr[:nbtest],idxto[:nbtest])
        txt += str(result)+'\n'
        for cc in listconcept:
            txt+='\n'
            txt += getnclosest(nbrank, idx2concept, concept2def, Esim, concept2idx[cc], 0, lhs = True, emb = True)
            for rr in listrel:
                txt+='\n'
                txt += getnclosest(nbrank, idx2concept, concept2def, sl, concept2idx[cc], concept2idx[rr], lhs = True, emb = False)
                txt+='\n'
                txt += getnclosest(nbrank, idx2concept, concept2def, sr, concept2idx[cc], concept2idx[rr], lhs = False, emb = False)
        f = open(savepath+'/model.pkl','w')
        cPickle.dump(embeddings,f,-1)
        cPickle.dump(leftop,f,-1)
        cPickle.dump(rightop,f,-1)
        f.close()
        f = open(savepath+'/currentrel.txt','w')
        f.write(txt)
        f.close()
        print result

