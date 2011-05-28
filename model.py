import theano
import theano.tensor as T
import numpy
import cPickle

# Similarity functions ----------------------------
def L1sim(left,right):
    return -T.sum(T.sqrt(T.sqr(left-right)),axis=1)

def L2sim(left,right):
    return -T.sqrt(T.sum(T.sqr(left-right),axis=1))

def dotsim(left,right):
    return T.sum(left*right,axis=1)

# -------------------------------------------------

# Costs -------------------------------------------
def margincost(pos,neg,marge=1.0):
    out = neg - pos + marge
    return T.sum(out * (out>0)),out>0

def validcost(pos,neg):
    out = neg - pos
    return T.sum(out * (out>0)),out>0, T.sum(out * (out<0))

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

def SimilarityFunctionl(fnsim,embeddings,leftop,rightop):
    idxrel = theano.sparse.csr_matrix('idxrel')
    idxright = theano.sparse.csr_matrix('idxright')
    idxleft = theano.sparse.csr_matrix('idxleft')
    lhs = (theano.sparse.dot(embeddings.E,idxleft).T).reshape((1,embeddings.D))
    rhs = (theano.sparse.dot(embeddings.E,idxright).T).reshape((1,embeddings.D))
    rel = (theano.sparse.dot(embeddings.E,idxrel).T).reshape((1,embeddings.D))
    simi = fnsim(leftop(lhs,rel),rightop(rhs,rel))
    return theano.function([idxleft,idxright,idxrel],[simi])

def SimilarityFunctionrightl(fnsim,embeddings,leftop,rightop,subtensorspec = None, adding = False):
    idxrel = theano.sparse.csr_matrix('idxrel')
    idxleft = theano.sparse.csr_matrix('idxleft')
    lhs = (theano.sparse.dot(embeddings.E,idxleft).T).reshape((1,embeddings.D))
    if not adding:
        if subtensorspec == None:
            rhs = embeddings.E.T
        else:
            rhs = embeddings.E[:,:subtensorspec].T
    else:
        idxadd = theano.sparse.csr_matrix('idxadd')
        sc = T.scalar('sc')
        if subtensorspec == None:
            rhs = embeddings.E.T * sc + (theano.sparse.dot(embeddings.E,idxadd).T).reshape((1,embeddings.D))
        else:
            rhs = embeddings.E[:,:subtensorspec].T * sc + (theano.sparse.dot(embeddings.E,idxadd).T).reshape((1,embeddings.D))
    rel = (theano.sparse.dot(embeddings.E,idxrel).T).reshape((1,embeddings.D))
    simi = fnsim(leftop(lhs,rel),rightop(rhs,rel))
    if not adding:
        return theano.function([idxleft,idxrel],[simi])
    else:
        return theano.function([idxleft,idxrel,idxadd,sc],[simi])

def SimilarityFunctionleftl(fnsim,embeddings,leftop,rightop,subtensorspec = None, adding = False):
    idxrel = theano.sparse.csr_matrix('idxrel')
    idxright = theano.sparse.csr_matrix('idxright')
    rhs = (theano.sparse.dot(embeddings.E,idxright).T).reshape((1,embeddings.D))
    if not adding:
        if subtensorspec == None:
            lhs = embeddings.E.T
        else:
            lhs = embeddings.E[:,:subtensorspec].T
    else:
        idxadd = theano.sparse.csr_matrix('idxadd')
        sc = T.scalar('sc')
        if subtensorspec == None:
            lhs = embeddings.E.T * sc + (theano.sparse.dot(embeddings.E,idxadd).T).reshape((1,embeddings.D))
        else:
            lhs = embeddings.E[:,:subtensorspec].T * sc + (theano.sparse.dot(embeddings.E,idxadd).T).reshape((1,embeddings.D))
    rel = (theano.sparse.dot(embeddings.E,idxrel).T).reshape((1,embeddings.D))
    simi = fnsim(leftop(lhs,rel),rightop(rhs,rel))
    if not adding:
        return theano.function([idxright,idxrel],[simi])
    else:
        return theano.function([idxright,idxrel,idxadd,sc],[simi])

def SimilarityFunctionrell(fnsim,embeddings,leftop,rightop,subtensorspec = None, adding = False):
    idxright = theano.sparse.csr_matrix('idxright')
    idxleft = theano.sparse.csr_matrix('idxleft')
    lhs = (theano.sparse.dot(embeddings.E,idxleft).T).reshape((1,embeddings.D))
    if not adding:
        if subtensorspec == None:
            rel = embeddings.E.T
        else:
            rel = embeddings.E[:,:subtensorspec].T
    else:
        idxadd = theano.sparse.csr_matrix('idxadd')
        sc = T.scalar('sc')
        if subtensorspec == None:
            rel = embeddings.E.T * sc + (theano.sparse.dot(embeddings.E,idxadd).T).reshape((1,embeddings.D))
        else:
            rel = embeddings.E[:,:subtensorspec].T * sc + (theano.sparse.dot(embeddings.E,idxadd).T).reshape((1,embeddings.D))
    rhs = (theano.sparse.dot(embeddings.E,idxright).T).reshape((1,embeddings.D))
    simi = fnsim(leftop(lhs,rel),rightop(rhs,rel))
    if not adding:
        return theano.function([idxleft,idxright],[simi])
    else:
        return theano.function([idxleft,idxright,idxadd,sc],[simi])


def SimilarityFunction(fnsim,embeddings,leftop,rightop):
    idxrel = T.iscalar('idxrel')
    idxright = T.iscalar('idxright')
    idxleft = T.iscalar('idxleft')
    lhs = (embeddings.E[:,idxleft]).reshape((1,embeddings.D))
    rhs = (embeddings.E[:,idxright]).reshape((1,embeddings.D))
    rel = (embeddings.E[:,idxrel]).reshape((1,embeddings.D))
    simi = fnsim(leftop(lhs,rel),rightop(rhs,rel))
    return theano.function([idxleft,idxright,idxrel],[simi])

def SimilarityFunctionright(fnsim,embeddings,leftop,rightop,subtensorspec = None):
    idxrel = T.iscalar('idxrel')
    idxleft = T.iscalar('idxleft')
    lhs = (embeddings.E[:,idxleft]).reshape((1,embeddings.D))
    if subtensorspec != None:
        rhs = (embeddings.E[:,:subtensorspec]).T
    else:
        rhs = embeddings.E.T
    rel = (embeddings.E[:,idxrel]).reshape((1,embeddings.D))
    simi = fnsim(leftop(lhs,rel),rightop(rhs,rel))
    return theano.function([idxleft,idxrel],[simi])

def SimilarityFunctionleft(fnsim,embeddings,leftop,rightop,subtensorspec = None):
    idxrel = T.iscalar('idxrel')
    idxright = T.iscalar('idxright')
    rhs = (embeddings.E[:,idxright]).reshape((1,embeddings.D))
    if subtensorspec != None:
        lhs = (embeddings.E[:,:subtensorspec]).T
    else:
        lhs = embeddings.E.T
    rel = (embeddings.E[:,idxrel]).reshape((1,embeddings.D))
    simi = fnsim(leftop(lhs,rel),rightop(rhs,rel))
    return theano.function([idxright,idxrel],[simi])

def SimilarityFunctionrel(fnsim,embeddings,leftop,rightop,subtensorspec = None):
    idxright = T.iscalar('idxrel')
    idxleft = T.iscalar('idxleft')
    lhs = (embeddings.E[:,idxleft]).reshape((1,embeddings.D))
    rel = embeddings.E.T
    if subtensorspec != None:
        rel = (embeddings.E[:,:subtensorspec]).T
    else:
        rel = embeddings.E.T
    rhs = (embeddings.E[:,idxright]).reshape((1,embeddings.D))
    simi = fnsim(leftop(lhs,rel),rightop(rhs,rel))
    return theano.function([idxleft,idxright],[simi])




def getnclosest(N, idx2lemme, lemme2idx, idx2synset, synset2idx, synset2def, synset2concept, concept2synset, simfn, part1, part2, typ = 1, emb = False):
    idx1 = []
    str1 = []
    idx2 = []
    str2 = []
    vec1 = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,1),dtype=theano.config.floatX)
    for i in part1:
        if i in lemme2idx.keys():
            idx1 += [lemme2idx[i]]
            vec1[idx1[-1],0] += 1/float(len(part1))
            str1 += ['-'+i]
        elif i in synset2idx.keys():
            idx1 += [synset2idx[i]]
            vec1[idx1[-1],0] += 1/float(len(part1))
            str1 += ['-'+synset2concept[i]]
        else:
            idx1 += [synset2idx[concept2synset[i]]]
            vec1[idx1[-1],0] += 1/float(len(part1))
            str1 += ['-'+i]
    vec2=scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,1),dtype=theano.config.floatX)
    for i in part2:
        if i in lemme2idx.keys():
            idx2 += [lemme2idx[i]]
            vec2[idx2[-1],0] += 1/float(len(part2))
            str2 += ['-'+i]
        elif i in synset2idx.keys():
            idx2 += [synset2idx[i]]
            vec2[idx2[-1],0] += 1/float(len(part2))
            str2 += ['-'+synset2concept[i]]
        else:
            idx2 += [synset2idx[concept2synset[i]]]
            vec2[idx2[-1],0] += 1/float(len(part2))
            str2 += ['-'+i]
    ll = (simfn(vec1,vec2)[0]).flatten()
    llo = numpy.argsort(ll)[::-1]
    llt = ll[llo]
    tt = ''
    txt1 =''
    for i in str1:
        txt1 += i
    txt2 = ''
    for i in str2:
        txt2 += i 
    if emb:
        tt += 'Similar to: %s\n'%( txt1 )
    else:
        if typ == 1:
            tt += '???? %s %s\n'%( txt2, txt1 )
        elif typ == 2:
            tt += '%s %s ????\n'%( txt1, txt2 )
        elif typ == 3:
            tt += '%s ???? %s\n'%( txt1, txt2 )
    for i in range(N):
        if llo[i] in idx2lemme.keys():
            stro = idx2lemme[llo[i]]
        elif idx2synset[llo[i]][0] == '_':
            stro = llo[i]
        else:
            stro = synset2concept[idx2synset[llo[i]]] + ' : ' + synset2def[idx2synset[llo[i]]]
        tt += 'Rank %s %s %s\n'%(i+1,llt[i],stro)
    return tt

import theano.sparse
import scipy.sparse

def TrainFunction(fnsim,embeddings, leftop, rightop, marge = 1.0, relb = True):
    # inputs 
    inpposr = theano.sparse.csr_matrix()
    inpposl = theano.sparse.csr_matrix()
    inpposo = theano.sparse.csr_matrix()
    inpposln = theano.sparse.csr_matrix()
    inpposrn = theano.sparse.csr_matrix()
    inpposon = theano.sparse.csr_matrix()
    lrparams = T.scalar('lrparams')
    lrembeddings = T.scalar('lrembeddings')
    # graph
    lhs = theano.sparse.dot(embeddings.E,inpposl).T
    rhs = theano.sparse.dot(embeddings.E,inpposr).T
    rel = theano.sparse.dot(embeddings.E,inpposo).T
    lhsn = theano.sparse.dot(embeddings.E,inpposln).T
    rhsn = theano.sparse.dot(embeddings.E,inpposrn).T
    reln = theano.sparse.dot(embeddings.E,inpposon).T
    simi = fnsim(leftop(lhs,rel),rightop(rhs,rel))
    siminl = fnsim(leftop(lhsn,rel),rightop(rhs,rel))
    siminr = fnsim(leftop(lhs,rel),rightop(rhsn,rel))
    simino = fnsim(leftop(lhs,reln),rightop(rhs,reln))
    costl,outl = margincost(simi,siminl,marge)
    costr,outr = margincost(simi,siminr,marge)
    costo,outo = margincost(simi,simino,marge)
    if relb:
        cost = costl + costr + costo
    else:
        cost = costl + costr
    out = T.concatenate([outl,outr,outo])
    if hasattr(fnsim,'params'):
        gradientsparams = T.grad(cost, leftop.params + rightop.params + fnsim.params)
        updates = dict((i,i-lrparams*j) for i,j in zip(leftop.params + rightop.params + fnsim.params, gradientsparams))
    else:
        gradientsparams = T.grad(cost, leftop.params + rightop.params)
        updates = dict((i,i-lrparams*j) for i,j in zip(leftop.params + rightop.params, gradientsparams))
    gradientsembeddings = T.grad(cost, embeddings.E)
    newE = embeddings.E - lrembeddings * gradientsembeddings
    ############### scaling variants
    #updates = dict((i,i-lrparams/(1+T.cast(T.sum(out),dtype=theano.config.floatX))*j) for i,j in zip(leftop.params + rightop.params, gradientsparams))
    #maskE = T.vector('maskE')
    #newE = (embeddings.E - lrembeddings/(1+maskE*T.cast(T.sum(out),dtype=theano.config.floatX)) * gradientsembeddings)
    ###############
    #newEnorm = newE / T.sqrt(T.sum(newE*newE,axis=0))
    updates.update({embeddings.E:newE})
    return theano.function([lrparams,lrembeddings,inpposl, inpposr, inpposo, inpposln, inpposrn,inpposon], [cost,costl,costr,costo,T.sum(out),T.sum(outl),T.sum(outr),T.sum(outo),lhs,rhs,rel,simi,siminl,siminr,simino],updates=updates)

def BatchSimilarityFunction(fnsim,embeddings, leftop, rightop):
    # inputs
    inpposr = theano.sparse.csr_matrix()
    inpposl = theano.sparse.csr_matrix()
    inpposo = theano.sparse.csr_matrix()
    # graph
    lhs = theano.sparse.dot(embeddings.E,inpposl).T
    rhs = theano.sparse.dot(embeddings.E,inpposr).T
    rel = theano.sparse.dot(embeddings.E,inpposo).T
    simi = fnsim(leftop(lhs,rel),rightop(rhs,rel))
    return theano.function([inpposl, inpposr, inpposo], [simi])


def BatchValidFunction(fnsim,embeddings, leftop, rightop):
    # inputs
    inpposr = theano.sparse.csr_matrix()
    inpposl = theano.sparse.csr_matrix()
    inpposo = theano.sparse.csr_matrix()
    inpposln = theano.sparse.csr_matrix()
    inpposrn = theano.sparse.csr_matrix()
    inpposon = theano.sparse.csr_matrix()
    # graph
    lhs = theano.sparse.dot(embeddings.E,inpposl).T
    rhs = theano.sparse.dot(embeddings.E,inpposr).T
    rel = theano.sparse.dot(embeddings.E,inpposo).T
    lhsn = theano.sparse.dot(embeddings.E,inpposln).T
    rhsn = theano.sparse.dot(embeddings.E,inpposrn).T
    reln = theano.sparse.dot(embeddings.E,inpposon).T
    simi = fnsim(leftop(lhs,rel),rightop(rhs,rel))
    siminl = fnsim(leftop(lhsn,rel),rightop(rhs,rel))
    siminr = fnsim(leftop(lhs,rel),rightop(rhsn,rel))
    simino = fnsim(leftop(lhs,reln),rightop(rhs,reln))
    costl,outl,margel = validcost(simi,siminl)
    costr,outr,marger = validcost(simi,siminr)
    costo,outo,margeo = validcost(simi,simino)
    cost = costl + costr + costo
    out = T.concatenate([outl,outr,outo])
    return theano.function([inpposl, inpposr, inpposo, inpposln, inpposrn,inpposon], [cost,costl,costr,costo,T.sum(out),T.sum(outl),T.sum(outr),T.sum(outo),margel,marger,margeo,lhs,rhs,rel,simi,siminl,siminr,simino])



def calctestval(sl,sr,idxtl,idxtr,idxto):
    errl = []
    errr = []
    for l,o,r in zip(idxtl,idxto,idxtr):
        errl += [numpy.argsort(numpy.argsort((sl(r,o)[0]).flatten())[::-1]).flatten()[l]]
        errr += [numpy.argsort(numpy.argsort((sr(l,o)[0]).flatten())[::-1]).flatten()[r]]
    return numpy.mean(errl+errr),numpy.std(errl+errr),numpy.mean(errl),numpy.std(errl),numpy.mean(errr),numpy.std(errr)


def calctestscore(sl,sr,so,posl,posr,poso):
    errl = []
    errr = []
    erro = []
    for i in range(posl.shape[1]):
        rankl = numpy.argsort((sl(posr[:,i],poso[:,i])[0]).flatten())
        for l in posl[:,i].nonzero()[0]:
            errl += [numpy.argsort(rankl[::-1]).flatten()[l]]
        rankr = numpy.argsort((sr(posl[:,i],poso[:,i])[0]).flatten())
        for r in posr[:,i].nonzero()[0]:
            errr += [numpy.argsort(rankr[::-1]).flatten()[r]]
        ranko = numpy.argsort((so(posl[:,i],posr[:,i])[0]).flatten())
        for o in poso[:,i].nonzero()[0]:
            erro += [numpy.argsort(ranko[::-1]).flatten()[0]]
    return numpy.mean(errl+errr+erro),numpy.std(errl+errr+erro),numpy.mean(errl),numpy.std(errl),numpy.mean(errr),numpy.std(errr),numpy.mean(erro),numpy.std(erro)

import copy

def calctestscore2(sl,sr,so,posl,posr,poso):
    errl = []
    errr = []
    erro = []
    for i in range(posl.shape[1]):
        lnz = posl[:,i].nonzero()[0]
        for j in lnz:
            val = posl[j,i]
            tmpadd = copy.deepcopy(posl[:,i])
            tmpadd[j,0] = 0.0
            rankl = numpy.argsort((sl(posr[:,i],poso[:,i],tmpadd,val)[0]).flatten())
            errl += [numpy.argsort(rankl[::-1]).flatten()[j]]
        rnz = posr[:,i].nonzero()[0]
        for j in rnz:
            val = posr[j,i]
            tmpadd = copy.deepcopy(posr[:,i])
            tmpadd[j,0] = 0.0
            rankr = numpy.argsort((sr(posl[:,i],poso[:,i],tmpadd,val)[0]).flatten())
            errr += [numpy.argsort(rankr[::-1]).flatten()[j]]
        onz = poso[:,i].nonzero()[0]
        for j in onz:
            val = poso[j,i]
            tmpadd = copy.deepcopy(poso[:,i])
            tmpadd[j,0] = 0.0
            ranko = numpy.argsort((so(posl[:,i],posr[:,i],tmpadd,val)[0]).flatten())
            erro += [numpy.argsort(ranko[::-1]).flatten()[j]]
    return numpy.mean(errl+errr+erro),numpy.std(errl+errr+erro),numpy.mean(errl),numpy.std(errl),numpy.mean(errr),numpy.std(errr),numpy.mean(erro),numpy.std(erro)


def calctestscore3(sl,sr,so,posl,posr,poso,poslc,posrc,posoc):
    errl = []
    errr = []
    erro = []
    for i in range(posl.shape[1]):
        lnz = posl[:,i].nonzero()[0]
        for j in lnz:
            val = posl[j,i]
            tmpadd = copy.deepcopy(posl[:,i])
            tmpadd[j,0] = 0.0
            rankl = numpy.argsort((sl(posr[:,i],poso[:,i],tmpadd,val)[0]).flatten())
            errl += [numpy.argsort(rankl[::-1]).flatten()[poslc[j,i]]]
        rnz = posr[:,i].nonzero()[0]
        for j in rnz:
            val = posr[j,i]
            tmpadd = copy.deepcopy(posr[:,i])
            tmpadd[j,0] = 0.0
            rankr = numpy.argsort((sr(posl[:,i],poso[:,i],tmpadd,val)[0]).flatten())
            errr += [numpy.argsort(rankr[::-1]).flatten()[posrc[j,i]]]
        onz = poso[:,i].nonzero()[0]
        for j in onz:
            val = poso[j,i]
            tmpadd = copy.deepcopy(poso[:,i])
            tmpadd[j,0] = 0.0
            ranko = numpy.argsort((so(posl[:,i],posr[:,i],tmpadd,val)[0]).flatten())
            erro += [numpy.argsort(ranko[::-1]).flatten()[posoc[j,i]]]
    return numpy.mean(errl+errr+erro),numpy.std(errl+errr+erro),numpy.mean(errl),numpy.std(errl),numpy.mean(errr),numpy.std(errr),numpy.mean(erro),numpy.std(erro)
