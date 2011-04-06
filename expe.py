import scipy.sparse
import cPickle
from model import *

train = 'train'
operator = 'quad'
ndim = 50
nbatches = 10
lrparam = 1
lremb = 0.01
nbtest = 50 
testall = 5


idx2concept = cPickle.load(open('idx2concept.pkl','r'))
concept2idx = cPickle.load(open('concept2idx.pkl','r'))
concept2def = cPickle.load(open('concept2def.pkl','r'))


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
random = scipy.sparse.lil_matrix((len(idx2concept.keys()),2*posl.shape[1]),dtype=theano.config.floatX)
idxr = numpy.asarray(numpy.random.randint(len(idx2concept.keys()),size=(2*posl.shape[1])),dtype='int32')
for idx,i in enumerate(idxr):
    random[i,idx]=1

random = random.tocsr()

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
embeddings = Embedd(numpy.random,len(idx2concept.keys()),ndim)

# train function
ft = TrainFunction(embeddings,leftop,rightop)

# simi function
sl = SimilarityFunctionleft(embeddings,leftop,rightop)
sr = SimilarityFunctionright(embeddings,leftop,rightop)

ct = 0
M = posl.shape[1]/nbatches
while 1:
    for i in range(nbatches):
        resl = ft(lrparam/float(M),lremb,posl[:,i*M:(i+1)*M],posr[:,i*M:(i+1)*M],poso[:,i*M:(i+1)*M],random[:,2*i*M:(2*i+1)*M],random[:,(2*i+1)*M:(2*i+2)*M])
        print resl[0]/float(2*M),resl[1]/float(M),resl[2]/float(M),resl[3]/float(2*M),resl[4]/float(M),resl[5]/float(M)
        embeddings.norma()
    order = numpy.random.permutation(posl.shape[1])
    posl,posr,poso = (posl[:,order],posr[:,order],poso[:,order])
    random = random[:,numpy.random.permutation(random.shape[1])]
    ct = ct + 1
    if ct/float(testall) == ct / testall:
        print '------ Epoch ', ct
        result = calctestval(sl,sr,idxtl[:nbtest],idxtr[:nbtest],idxto[:nbtest])
        print result

