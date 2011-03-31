import theano
import theano.tensor as T
import numpy
import cPickle

# Similarity functions ----------------------------
def L1sim(left,right):
    return -T.sum(T.abs_(left-right),axis=1)

def dotsim(left,right):
    return T.sum(left*right,axis=1)
# -------------------------------------------------

# Costs -------------------------------------------
def margincost(pos,neg):
    out = neg - pos + 1.0
    return T.mean(out * (out>0))
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
    def __init__(self,N):
        self.N=N

    def __call__(self,x):
        return x[:,:self.N]

    def save(self,path):
        f = open(path,'w')
        cPickle.dump(self,f,-1)
        f.close()

class Embedd(object):
    def __init__(self,rng,N,D,Einit = None):
        self.N = N
        self.D = D
        if Einit == None:
            wbound = numpy.sqrt(6)
            W_values = numpy.asarray( rng.uniform( low = -wbound, high = wbound, \
                                    size = (N, D)), dtype = theano.config.floatX)
            self.E = theano.shared(value = W_values, name = 'E')
    
    def normalize(self):
        self.E.value /= numpy.sum(self.E.value * self.E.value,axis=1)


# ---------------------------------------

def SimilarityFunction(embeddings,leftop,rightop):
    idxrel = T.iscalar('idxrel')
    idxright = T.iscalar('idxright')
    idxleft = T.iscalar('idxleft')
    lhs = (embeddings.E[idxleft]).reshape((1,embeddings.D))
    rhs = (embeddings.E[idxright]).reshape((1,embeddings.D))
    rel = (embeddings.E[idxrel]).reshape((1,embeddings.D))
    simi = L1sim(leftop(lhs,rel),rightop(rhs,rel))
    return theano.function([idxleft,idxright,idxrel],[simi])

def SimilarityFunctionright(embeddings,leftop,rightop):
    idxrel = T.iscalar('idxrel')
    idxleft = T.iscalar('idxleft')
    lhs = (embeddings.E[idxleft]).reshape((1,embeddings.D))
    rhs = embeddings.E
    rel = (embeddings.E[idxrel]).reshape((1,embeddings.D))
    simi = L1sim(leftop(lhs,rel),rightop(rhs,rel))
    return theano.function([idxleft,idxrel],[simi])

def SimilarityFunctionleft(embeddings,leftop,rightop):
    idxrel = T.iscalar('idxrel')
    idxright = T.iscalar('idxleft')
    rhs = (embeddings.E[idxright]).reshape((1,embeddings.D))
    lhs = embeddings.E
    rel = (embeddings.E[idxrel]).reshape((1,embeddings.D))
    simi = L1sim(leftop(lhs,rel),rightop(rhs,rel))
    return theano.function([idxright,idxrel],[simi])

#def SimilarityFunction(embeddings,leftop,rightop):
#    idxrel = T.ivector('idxrel')
#    idxright = T.ivector('idxright')
#    idxleft = T.ivector('idxleft')
#    lhs = (embeddings.E[idxleft]).reshape((10,embeddings.D))
#    rhs = (embeddings.E[idxright]).reshape((10,embeddings.D))
#    rel = (embeddings.E[idxrel]).reshape((10,embeddings.D))
#    simi = L1sim(leftop(lhs,rel),rightop(rhs,rel))
#    return theano.function([idxleft,idxright,idxrel],[simi])

def SimilarityFunctionrel(embeddings,leftop,rightop):
    idxright = T.iscalar('idxrel')
    idxleft = T.iscalar('idxleft')
    lhs = (embeddings.E[idxleft]).reshape((1,embeddings.D))
    rel = embeddings.E
    lhs = (embeddings.E[idxright]).reshape((1,embeddings.D))
    simi = L1sim(leftop(lhs,rel),rightop(rhs,rel))
    return theano.function([idxleft,idxright],[simi])

def TrainFunctionright(embeddings,leftop,rightop):
    idxrel = T.iscalar('idxrel')
    idxright = T.iscalar('idxright')
    idxleft = T.iscalar('idxleft')
    idxrightn = T.iscalar('idxrightn')
    lhs = (embeddings.E[idxleft]).reshape((1,embeddings.D))
    rhs = (embeddings.E[idxright]).reshape((1,embeddings.D))
    rel = (embeddings.E[idxrel]).reshape((1,embeddings.D))
    simi = L1sim(leftop(lhs,rel),rightop(rhs,rel))
    rhsn = (embeddings.E[idxrightn]).reshape((1,embeddings.D))
    simin = L1sim(leftop(lhs,rel),rightop(rhsn,rel))
    cost = margincost(simi,simin)
    gradientsparams = T.grad(cost, leftop.params + rightop.params)
    gradientembd = T.grad(cost, [lhs,rhs,rel])
    updates = dict((i,i-j) for i,j in zip(leftop.params + rightop.params, gradientsparams))
    #updates.update(dict((i,i-j) for i,j in zip([embeddings.E[idxleftn],embeddings.E[idxrightn],embeddings.E[idxreln]], gradientembd)))
    return theano.function([idxleft,idxright,idxrel,idxrightn],[cost]+gradientembd,updates = updates)

def TrainFunctionleft(embeddings,leftop,rightop):
    idxrel = T.iscalar('idxrel')
    idxright = T.iscalar('idxright')
    idxleft = T.iscalar('idxleft')
    idxrightn = T.iscalar('idxrightn')
    lhs = (embeddings.E[idxleft]).reshape((1,embeddings.D))
    rhs = (embeddings.E[idxright]).reshape((1,embeddings.D))
    rel = (embeddings.E[idxrel]).reshape((1,embeddings.D))
    simi = L1sim(leftop(lhs,rel),rightop(rhs,rel))
    rhsn = (embeddings.E[idxrightn]).reshape((1,embeddings.D))
    simin = L1sim(leftop(lhs,rel),rightop(rhsn,rel))
    cost = margincost(simi,simin)
    gradientsparams = T.grad(cost, leftop.params + rightop.params)
    gradientembd = T.grad(cost, [lhs,rhs,rel])
    updates = dict((i,i-j) for i,j in zip(leftop.params + rightop.params, gradientsparams))
    #updates.update(dict((i,i-j) for i,j in zip([embeddings.E[idxleftn],embeddings.E[idxrightn],embeddings.E[idxreln]], gradientembd)))
    return theano.function([idxleft,idxright,idxrel,idxleftn],[cost]+gradientembd,updates = updates)


def TrainFunction(nbbatchpos, embeddings, leftop, rightop):
    posr = T.ivector('posr')
    posl = T.ivector('posl')
    poso = T.ivector('poso')
    posln = T.ivector('posln')
    lhs = T.concatenate([embeddings.E[posl[i]].reshape((1,embeddings.D)) for i in range(nbbatchpos)],axis=0)
    rhs = T.concatenate([embeddings.E[posr[i]].reshape((1,embeddings.D)) for i in range(nbbatchpos)],axis=0)
    rel = T.concatenate([embeddings.E[poso[i]].reshape((1,embeddings.D)) for i in range(nbbatchpos)],axis=0)
    lhsn = T.concatenate([embeddings.E[posln[i]].reshape((1,embeddings.D)) for i in range(nbbatchpos)],axis=0)
    print lhs,lhs.ndim
    print leftop(lhs,rel).ndim
    simi = L1sim(leftop(lhs,rel),rightop(rhs,rel))
    simin = L1sim(leftop(lhsn,rel),rightop(rhs,rel))
    cost = margincost(simi,simin)
    gradientsparams = T.grad(cost, leftop.params + rightop.params)
    gradientembd = T.grad(cost, [lhs,rhs,rel])
    updates = dict((i,i-0.01*j) for i,j in zip(leftop.params + rightop.params, gradientsparams))
    return theano.function([posr, posl, poso, posln], [cost]+gradientembd,updates = updates)


embeddings = Embedd(numpy.random,90000,50)
#leftop = MLP(numpy.random, 'rect', 50, 50, 75, 50)   
#rightop = MLP(numpy.random, 'rect', 50, 50, 75, 50)
leftop = Quadlayer(numpy.random, 50, 50, 75, 50)   
rightop = Quadlayer(numpy.random, 50, 50, 75, 50)
f = SimilarityFunction(embeddings,leftop,rightop)

print 'coucou'
import time

#for i in range(1):
#    print f(numpy.random.randint(90000,size=(10,)),numpy.random.randint(90000,size=(10,)),numpy.random.randint(90000),size=(10,))[0].shape

#print (time.time() - tt)

f = TrainFunction(1000,embeddings,leftop,rightop)

c = [numpy.random.randint(90000,size=(1000)) for i in range(4)]
tt = time.time()
f(*c)
print (time.time() - tt)



#tt = time.time()
#for i in range(1000):
#    print f(numpy.random.randint(90000),numpy.random.randint(90000),numpy.random.randint(90000),numpy.random.randint(90000))




# create function that associate embedig + left + right operator + cost to give similarity and train system

