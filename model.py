import theano
import theano.tensor as T
import numpy
import cPickle

# Similarity functions ----------------------------
def L1sim(left,right):
    return -T.sum(T.sqrt(T.sqr(left-right)),axis=1)

def L2sim(left,right):
    return -T.sqrt(T.sum(T.sqr(left-right)),axis=1)

def dotsim(left,right):
    return T.sum(left*right,axis=1)

# -------------------------------------------------

# Costs -------------------------------------------
def margincost(pos,neg):
    out = neg - pos + 1.
    return T.sum(out * (out>0)),out>0

# -------------------------------------------------

# Activation functions ----------------------------
def htanh(x):
    return -1. * (x<-1.) + x * (x<1.) * (x>=-1.) + 1. * (x>=1)

def hsigm(x):
    return x * (x<1) * (x>0)  + 1. * (x>=1)

def rect(x):
    return x*(x>0)

def sigm(x):
    return T.nnet.sigmoid(x)

def tanh(x):
    return T.tanh(x)

def lin(x):
    return x

# -------------------------------------------------


# Layers ------------------------------------------
class Layer(object):
    def __init__(self, rng, act, n_inp, n_out, Winit = None, tag=''):
        self.act = eval(act)
        self.actstr = act
        self.n_inp = n_inp
        self.n_out = n_out
        # init param
        if Winit == None:
            wbound = numpy.sqrt(6./(n_inp+n_out))
            W_values = numpy.asarray( rng.uniform( low = -wbound, high = wbound, \
                                    size = (n_inp, n_out)), dtype = theano.config.floatX)
            self.W = theano.shared(value = W_values, name = 'W'+tag)
        else:
            self.W = theano.shared(value = Winit, name = 'W'+tag)
        self.params = [self.W]
    def __call__(self,x):
        return self.act(T.dot(x, self.W))
    def save(self,path):
        f = open(path,'w')
        cPickle.dump(self,f,-1)
        f.close()

class Layercomb(object):
    def __init__(self, rng, act, n_inp1, n_inp2 , n_out, W1init = None, W2init = None, binit = None):
        self.act = eval(act)
        self.actstr = act
        self.n_inp1 = n_inp1
        self.n_inp2 = n_inp2
        self.n_out = n_out
        self.layer1 = Layer(rng, 'lin', n_inp1, n_out, Winit = W1init, tag = '1')
        self.layer2 = Layer(rng, 'lin', n_inp2, n_out, Winit = W2init, tag = '2')
        if binit == None:
            b_values = numpy.zeros((n_out,), dtype= theano.config.floatX)
            self.b = theano.shared(value= b_values, name = 'b')
        else:
            self.b = theano.shared(value = binit, name = 'b')
        self.params = self.layer1.params + self.layer2.params + [self.b]
    def __call__(self,x,y):
        return self.act(T.dot(x, self.layer1.W) + T.dot(y, self.layer2.W) + self.b)
    def save(self,path):
        f = open(path,'w')
        cPickle.dump(self,f,-1)
        f.close()


class MLP(object):
    def __init__(self, rng, act, n_inp1, n_inp2, n_hid, n_out, W1init = None, W2init = None, b12init = None, W3init = None, b3init = None):
        self.act = eval(act)
        self.actstr = act
        self.n_inp1 = n_inp1
        self.n_inp2 = n_inp2
        self.n_hid = n_hid
        self.n_out = n_out
        self.layer12 = Layercomb(rng, act, n_inp1, n_inp2, n_hid, W1init = W1init, W2init = W2init, binit = b12init)
        self.layer3 = Layer(rng, 'lin', n_hid, n_out, Winit = W3init, tag = '3')
        if b3init == None:
            b_values = numpy.zeros((n_out,), dtype= theano.config.floatX)
            self.b = theano.shared(value= b_values, name = 'b')
        else:
            self.b = theano.shared(value = b3init, name = 'b')
        self.params = self.layer12.params + self.layer3.params + [self.b]
    def __call__(self,x,y):
        return self.layer3(self.layer12(x,y))
    def save(self,path):
        f = open(path,'w')
        cPickle.dump(self,f,-1)
        f.close()

class Quadlayer(object):
    def __init__(self, rng, n_inp1, n_inp2, n_hid, n_out, W1init = None, b1init = None, W2init = None, b2init = None, W3init = None, b3init = None):
        self.n_inp1 = n_inp1
        self.n_inp2 = n_inp2
        self.n_hid = n_hid
        self.n_out = n_out
        if W1init == None:
            wbound = numpy.sqrt(6./(n_inp1+n_hid))
            W_values = numpy.asarray( rng.uniform( low = -wbound, high = wbound, \
                                    size = (n_inp1, n_hid)), dtype = theano.config.floatX)
            self.W1 = theano.shared(value = W_values, name = 'W1')
        else:
            self.W1 = theano.shared(value = W1init, name = 'W1')
        if b1init == None:
            b_values = numpy.zeros((n_hid,), dtype= theano.config.floatX)
            self.b1 = theano.shared(value= b_values, name = 'b1')
        else:
            self.b1 = theano.shared(value = b1init, name = 'b1')
        if W2init == None:
            wbound = numpy.sqrt(6./(n_inp2+n_hid))
            W_values = numpy.asarray( rng.uniform( low = -wbound, high = wbound, \
                                    size = (n_inp2, n_hid)), dtype = theano.config.floatX)
            self.W2 = theano.shared(value = W_values, name = 'W2')
        else:
            self.W2 = theano.shared(value = W2init, name = 'W2')
        if b2init == None:
            b_values = numpy.zeros((n_hid,), dtype= theano.config.floatX)
            self.b2 = theano.shared(value= b_values, name = 'b2')
        else:
            self.b2 = theano.shared(value = b2init, name = 'b2')
        if W3init == None:
            wbound = numpy.sqrt(6./(n_hid+n_out))
            W_values = numpy.asarray( rng.uniform( low = -wbound, high = wbound, \
                                    size = (n_hid, n_out)), dtype = theano.config.floatX)
            self.W3 = theano.shared(value = W_values, name = 'W3')
        else:
            self.W3 = theano.shared(value = W3init, name = 'W3')
        if b3init == None:
            b_values = numpy.zeros((n_out,), dtype= theano.config.floatX)
            self.b3 = theano.shared(value= b_values, name = 'b3')
        else:
            self.b3 = theano.shared(value = b3init, name = 'b3')
        self.params = [self.W1,self.b1,self.W2,self.b2,self.W3,self.b3]
    def __call__(self,x,y):
        return T.dot((T.dot(x,self.W1) + self.b1) * (T.dot(y,self.W2) + self.b2), self.W3 ) + self.b3
    def save(self,path):
        f = open(path,'w')
        cPickle.dump(self,f,-1)
        f.close()

class Id(object):
    def __init__(self):
        self.params = []
    def __call__(self,x,y):
        return x
    def save(self,path):
        pass

class Embedd(object):
    def __init__(self,rng,N,D,Einit = None):
        self.N = N
        self.D = D
        if Einit == None:
            wbound = numpy.sqrt(6)
            W_values = numpy.asarray( rng.uniform( low = -wbound, high = wbound, \
                                    size = (D, N)), dtype = theano.config.floatX)
            self.E = theano.shared(value = W_values/numpy.sqrt(numpy.sum(W_values * W_values,axis=0)), name = 'E')
        self.updates = {self.E:self.E/T.sqrt(T.sum(self.E * self.E,axis=0))}
        self.norma = theano.function([],[],updates = self.updates)
    def normalize(self):
        self.norma()


# ---------------------------------------

def SimilarityFunction(embeddings,leftop,rightop):
    idxrel = T.iscalar('idxrel')
    idxright = T.iscalar('idxright')
    idxleft = T.iscalar('idxleft')
    lhs = (embeddings.E[:,idxleft]).reshape((1,embeddings.D))
    rhs = (embeddings.E[:,idxright]).reshape((1,embeddings.D))
    rel = (embeddings.E[:,idxrel]).reshape((1,embeddings.D))
    simi = dotsim(leftop(lhs,rel),rightop(rhs,rel))
    return theano.function([idxleft,idxright,idxrel],[simi])

def SimilarityFunctionright(embeddings,leftop,rightop):
    idxrel = T.iscalar('idxrel')
    idxleft = T.iscalar('idxleft')
    lhs = (embeddings.E[:,idxleft]).reshape((1,embeddings.D))
    rhs = embeddings.E.T
    rel = (embeddings.E[:,idxrel]).reshape((1,embeddings.D))
    simi = dotsim(leftop(lhs,rel),rightop(rhs,rel))
    return theano.function([idxleft,idxrel],[simi])

def SimilarityFunctionleft(embeddings,leftop,rightop):
    idxrel = T.iscalar('idxrel')
    idxright = T.iscalar('idxleft')
    rhs = (embeddings.E[:,idxright]).reshape((1,embeddings.D))
    lhs = embeddings.E.T
    rel = (embeddings.E[:,idxrel]).reshape((1,embeddings.D))
    simi = dotsim(leftop(lhs,rel),rightop(rhs,rel))
    return theano.function([idxright,idxrel],[simi])

def SimilarityFunctionrel(embeddings,leftop,rightop):
    idxright = T.iscalar('idxrel')
    idxleft = T.iscalar('idxleft')
    lhs = (embeddings.E[:,idxleft]).reshape((1,embeddings.D))
    rel = embeddings.E.T
    lhs = (embeddings.E[:,idxright]).reshape((1,embeddings.D))
    simi = dotsim(leftop(lhs,rel),rightop(rhs,rel))
    return theano.function([idxleft,idxright],[simi])

#def getnclosest(simf,idx1,dix2

import theano.sparse
import scipy.sparse

def TrainFunction(embeddings, leftop, rightop):
    # inputs 
    inpposr = theano.sparse.csr_matrix()
    inpposl = theano.sparse.csr_matrix()
    inpposo = theano.sparse.csr_matrix()
    inpposln = theano.sparse.csr_matrix()
    inpposrn = theano.sparse.csr_matrix()
    lrparams = T.scalar('lrparams')
    lrembeddings = T.scalar('lrembeddings')
    # graph
    lhs = theano.sparse.dot(embeddings.E,inpposl).T
    rhs = theano.sparse.dot(embeddings.E,inpposr).T
    rel = theano.sparse.dot(embeddings.E,inpposo).T
    lhsn = theano.sparse.dot(embeddings.E,inpposln).T
    rhsn = theano.sparse.dot(embeddings.E,inpposrn).T
    simi = dotsim(leftop(lhs,rel),rightop(rhs,rel))
    siminl = dotsim(leftop(lhsn,rel),rightop(rhs,rel))
    siminr = dotsim(leftop(lhs,rel),rightop(rhsn,rel))
    costl,outl = margincost(simi,siminl)
    costr,outr = margincost(simi,siminr)
    cost = costl + costr
    out = T.concatenate([outl,outr])
    gradientsparams = T.grad(cost, leftop.params + rightop.params)
    gradientsembeddings = T.grad(cost, embeddings.E)
    ############### scaling variants
    #updates = dict((i,i-lrparams/(1+T.cast(T.sum(out),dtype=theano.config.floatX))*j) for i,j in zip(leftop.params + rightop.params, gradientsparams))
    #maskE = T.vector('maskE')
    #newE = (embeddings.E - lrembeddings/(1+maskE*T.cast(T.sum(out),dtype=theano.config.floatX)) * gradientsembeddings)
    ###############
    updates = dict((i,i-lrparams*j) for i,j in zip(leftop.params + rightop.params, gradientsparams))
    newE = embeddings.E - lrembeddings * gradientsembeddings
    #newEnorm = newE / T.sqrt(T.sum(newE*newE,axis=0))
    updates.update({embeddings.E:newE})
    return theano.function([lrparams,lrembeddings,inpposl, inpposr, inpposo, inpposln, inpposrn], [cost,costl,costr,T.sum(out),T.sum(outl),T.sum(outr),lhs,rhs,rel,simi,siminl,siminr],updates=updates)


def calctestval(sl,sr,idxtl,idxtr,idxto):
    errl = []
    errr = []
    for l,o,r in zip(idxtl,idxto,idxtr):
        errl += [numpy.argsort(numpy.argsort(sl(r,o))[:,::-1]).flatten()[l]]
        errr += [numpy.argsort(numpy.argsort(sr(l,o))[:,::-1]).flatten()[r]]
    return numpy.mean(errl+errr),numpy.std(errl+errr),numpy.mean(errl),numpy.std(errl),numpy.mean(errr),numpy.std(errr)
