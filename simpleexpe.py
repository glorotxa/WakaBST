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


def warpsampling(fft,posl,posr,poso,posln,posrn,poson,N):
    # This simple function does the warp sampling in a unefficient manner.
    # (take the batch, do the forward, resample negative elements associated to cost = 0, re-do the forward... etc N times)
    # the resampling is done by shuffling the negative indexes matrices.
    # fft is the forward function (returning the cost>0 vector)
    count_sample = 0
    nbup = 0
    while count_sample<N and nbup!=3*(posl.shape[1]):
        outl,outr,outo = fft(posl,posr,poso,posln,posrn,poson)
        newposln = copy.deepcopy(posln[:,numpy.random.permutation(posln.shape[1])])
        newposrn = copy.deepcopy(posrn[:,numpy.random.permutation(posrn.shape[1])])
        newposon = copy.deepcopy(poson[:,numpy.random.permutation(poson.shape[1])])
        ident1 = scipy.sparse.identity(posln.shape[1])
        ident1.data = outl
        ident2 = scipy.sparse.identity(posln.shape[1])
        ident2.data = 1-outl
        posln = posln * ident1 + newposln * ident2
        ident1 = scipy.sparse.identity(posln.shape[1])
        ident1.data = outr
        ident2 = scipy.sparse.identity(posln.shape[1])
        ident2.data = 1-outr
        posrn = posrn * ident1 + newposrn * ident2
        ident1 = scipy.sparse.identity(posln.shape[1])
        ident1.data = outo
        ident2 = scipy.sparse.identity(posln.shape[1])
        ident2.data = 1-outo
        poson = poson * ident1 + newposon * ident2
        count_sample += 1
        nbup = numpy.sum(outl) +  numpy.sum(outr) +  numpy.sum(outo)
    return posln,posrn,poson

synset2lemme = cPickle.load(open(datpath+'synset2lemme.pkl','r'))
lemme2synset = cPickle.load(open(datpath+'lemme2synset.pkl','r'))
lemme2freq = cPickle.load(open(datpath+'lemme2freq.pkl','r'))
synset2idx = cPickle.load(open(datpath+'synset2idx.pkl','r'))
idx2synset = cPickle.load(open(datpath+'idx2synset.pkl','r'))
lemme2idx = cPickle.load(open(datpath+'lemme2idx.pkl','r'))
idx2lemme = cPickle.load(open(datpath+'idx2lemme.pkl','r'))
synset2neg = cPickle.load(open(datpath+'synset2neg.pkl','r'))
synset2def = cPickle.load(open(datpath+'synset2def.pkl','r'))
synset2concept = cPickle.load(open(datpath+'synset2concept.pkl','r'))
concept2synset = cPickle.load(open(datpath+'concept2synset.pkl','r'))

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
random = scipy.sparse.lil_matrix((len(idx2concept.keys()+idx2amb.keys()),3*posl.shape[1]),dtype=theano.config.floatX)
idxr = numpy.asarray(numpy.random.randint(len(idx2concept.keys()+idx2amb.keys()),size=(3*posl.shape[1])),dtype='int32')
for idx,i in enumerate(idxr):
    random[i,idx]=1

random = random.tocsr()

if not loadmodel:
    # operators
    if  operator == 'Id':
        leftop = Id()
        rightop = Id()
    elif  operator == 'linear':
        leftop = Layercomb(numpy.random, 'lin', ndim, ndim, ndim)
        rightop = Layercomb(numpy.random, 'lin', ndim, ndim, ndim)
    elif  operator == 'mlp':
        leftop = MLP(numpy.random, 'sigm', ndim, ndim, (3*ndim)/2, ndim)
        rightop = MLP(numpy.random, 'sigm', ndim, ndim, (3*ndim)/2, ndim)
    elif  operator == 'quad':
        leftop = Quadlayer(numpy.random, ndim, ndim, (3*ndim)/2, ndim)
        rightop = Quadlayer(numpy.random, ndim, ndim, (3*ndim)/2, ndim)
    if simfnstr == 'MLP':
        MLPout = MLP(numpy.random, 'sigm', ndim, ndim, ndim, 1)
    # embeddings
    embeddings = Embedd(numpy.random,numpy.max(lemme2idx.values())+1,ndim)
else:
    f = open(loadmodel)
    embeddings = cPickle.load(f)
    leftop = cPickle.load(f)
    rightop = cPickle.load(f)
    if simfnstr == 'MLP':
       MLPout = cPickle.load(f)
    f.close()

if simfnstr == 'MLP':
    simfn = MLPout
else:
    simfn = eval(simfnstr+'sim')


# train function
ft = TrainFunction(simfn,embeddings,leftop,rightop,marge = 1., relb = True)
# relb = True : the relation is also sampled for the negative example.

# Forward Only
fftwn = ForwardFunction(simfn,embeddings,leftop,rightop, marge = 1.)

# Batch function
vt = BatchValidFunction(simfn,embeddings,leftop,rightop)



#-----------------------------
# Scoring functions creation:
#-----------------------------

# -- Scoring over indexes
# left
sl = SimilarityFunctionleft(simfn,embeddings,leftop,rightop,subtensorspec = numpy.max(synset2idx.values())+1)
# subtensorspec: define a subtensor from which do the ranking (here only on the synsets)
#right
sr = SimilarityFunctionright(simfn,embeddings,leftop,rightop,subtensorspec = numpy.max(synset2idx.values())+1)
# subtensorspec: define a subtensor from which do the ranking (here only on the synsets

# -- Scoring over batches (sparse matrices)
# right
srl = SimilarityFunctionrightl(simfn,embeddings,leftop,rightop)
# left
sll = SimilarityFunctionleftl(simfn,embeddings,leftop,rightop)
# relation
sol = SimilarityFunctionrell(simfn,embeddings,leftop,rightop)

# -- Distance between lhs and rhs embeddings
leftopid = Id()
rightopid = Id()
Esim = SimilarityFunctionright(L2sim,embeddings,leftopid,rightopid)


# Init. of variables for the training loop:
ct = 0
M = posl.shape[1]/nbatches
left = []
right = []
leftb = []
rightb = []
rel = []
relb = []


#-------------------------
# Training loop:
#------------------------

while 1:
    for i in range(nbatches):
        tmpnl = copy.deepcopy(random[:,3*i*M:(3*i+1)*M])
        tmpno = copy.deepcopy(random[:,(3*i+1)*M:(3*i+2)*M])
        tmpnr = copy.deepcopy(random[:,(3*i+2)*M:(3*i+3)*M])
        if warp:
            tmpnl,tmpnr,tmpno = warpsampling(fftwnl,posl[:,i*M:(i+1)*M],posr[:,i*M:(i+1)*M],poso[:,i*M:(i+1)*M],tmpnl,tmpnr,tmpno,warp)
        resl = ft(lrparam/float(M),lremb,posl[:,i*M:(i+1)*M],posr[:,i*M:(i+1)*M],poso[:,i*M:(i+1)*M],tmpnl,tmpnr,tmpno)
        left += [resl[1]/float(M)]
        right += [resl[2]/float(M)] 
        rel += [resl[3]/float(M)]
        leftb += [resl[5]/float(M)]
        rightb += [resl[6]/float(M)]
        relb += [resl[7]/float(M)]
        embeddings.norma() #normalize the embeddings
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
        rel = []
        relb = []
        result = calctestval(sl,sr,idxtl[:nbtest],idxtr[:nbtest],idxto[:nbtest])
        txt += str(result)+'\n'
        for cc in listconcept:
            txt+='\n'
            txt += getnclosest(nbrank, idx2lemme, lemme2idx, idx2synset, synset2idx, synset2def, synset2concept, concept2synset, Esim, cc, [], typ = 0, emb = True)
            for rr in listrel:
                txt+='\n'
                txt += getnclosest(nbrank,idx2lemme, lemme2idx, idx2synset, synset2idx, synset2def, synset2concept, concept2synset , sll, cc, rr, typ = 1, emb = False)
                txt+='\n'
                txt += getnclosest(nbrank,idx2lemme, lemme2idx, idx2synset, synset2idx, synset2def, synset2concept, concept2synset , srl, cc, rr, typ = 2, emb = False)
            for rr in listconcept:
                txt +='\n'
                txt += getnclosest(nbrank,idx2lemme, lemme2idx, idx2synset, synset2idx, synset2def, synset2concept, concept2synset , srl, cc, rr, typ = 3, emb = False)
        f = open(savepath+'/model.pkl','w')
        cPickle.dump(embeddings,f,-1)
        cPickle.dump(leftop,f,-1)
        cPickle.dump(rightop,f,-1)
        if simfnstr == 'MLP':
            cPickle.dump(MLPout,f,-1)
        f.close()
        f = open(savepath+'/currentrel.txt','w')
        f.write(txt)
        f.close()
        print result

