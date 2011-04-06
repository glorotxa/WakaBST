




def TrainFunctionleft(embeddings, leftop, rightop):
    inpposr = theano.sparse.csr_matrix()
    inpposl = theano.sparse.csr_matrix()
    inpposo = theano.sparse.csr_matrix()
    inpposln = theano.sparse.csr_matrix()
    lrparams = T.scalar('lrparams')
    lrembeddings = T.scalar('lrembeddings')
    lhs = theano.sparse.dot(embeddings.E,inpposl).T
    rhs = theano.sparse.dot(embeddings.E,inpposr).T
    rel = theano.sparse.dot(embeddings.E,inpposo).T
    lhsn = theano.sparse.dot(embeddings.E,inpposln).T
    simi = dotsim(leftop(lhs,rel),rightop(rhs,rel))
    simin = dotsim(leftop(lhsn,rel),rightop(rhs,rel))
    cost,out = margincost(simi,simin)
    gradientsparams = T.grad(cost, leftop.params + rightop.params)
    gradientsembeddings = T.grad(cost, embeddings.E)

    #updates = dict((i,i-lrparams/(1+T.cast(T.sum(out),dtype=theano.config.floatX))*j) for i,j in zip(leftop.params + rightop.params, gradientsparams))
    #maskE = T.vector('maskE')
    #newE = (embeddings.E - lrembeddings/(1+maskE*T.cast(T.sum(out),dtype=theano.config.floatX)) * gradientsembeddings)
    updates = dict((i,i-lrparams*j) for i,j in zip(leftop.params + rightop.params, gradientsparams))
    newE = embeddings.E - lrembeddings * gradientsembeddings
    #newEnorm = newE / T.sqrt(T.sum(newE*newE,axis=0))
    updates.update({embeddings.E:newE})
    return theano.function([lrparams,lrembeddings,inpposr, inpposl, inpposo, inpposln], [cost,lhs,rhs,rel,simi,simin,T.sum(out),cost,gradientsembeddings],updates=updates)
    #return theano.function([lrparams,lrembeddings,inpposr, inpposl, inpposo, inpposln,maskE], [cost,lhs,rhs,rel,simi,simin,T.sum(out),cost,gradientsembeddings,(1+T.cast(T.sum(out),dtype=theano.config.floatX))],updates=updates)
    #return theano.function([lrparams,lrembeddings,inpposr, inpposl, inpposo, inpposln], [cost],updates=updates)


def TrainFunctionright(embeddings, leftop, rightop):
    inpposr = theano.sparse.csr_matrix()
    inpposl = theano.sparse.csr_matrix()
    inpposo = theano.sparse.csr_matrix()
    inpposrn = theano.sparse.csr_matrix()
    lrparams = T.scalar('lrparams')
    lrembeddings = T.scalar('lrembeddings')
    lhs = theano.sparse.dot(embeddings.E,inpposl).T
    rhs = theano.sparse.dot(embeddings.E,inpposr).T
    rel = theano.sparse.dot(embeddings.E,inpposo).T
    rhsn = theano.sparse.dot(embeddings.E,inpposrn).T
    simi = dotsim(leftop(lhs,rel),rightop(rhs,rel))
    simin = dotsim(leftop(lhs,rel),rightop(rhsn,rel))
    cost,out = margincost(simi,simin)
    gradientsparams = T.grad(cost, leftop.params + rightop.params)
    gradientsembeddings = T.grad(cost, embeddings.E)
    # -------------------------------------------------
    #updates = dict((i,i-lrparams/(1+T.cast(T.sum(out),dtype=theano.config.floatX))*j) for i,j in zip(leftop.params + rightop.params, gradientsparams))
    #maskE = T.vector('maskE')
    #newE = (embeddings.E - lrembeddings/(1+maskE*T.cast(T.sum(out),dtype=theano.config.floatX)) * gradientsembeddings)
    # ------------------------------------------------
    updates = dict((i,i-lrparams*j) for i,j in zip(leftop.params + rightop.params, gradientsparams))
    newE = (embeddings.E - lrembeddings * gradientsembeddings)
    #newEnorm = newE / T.sqrt(T.sum(newE*newE,axis=0))
    updates.update({embeddings.E:newE})
    return theano.function([lrparams,lrembeddings,inpposr, inpposl, inpposo, inpposrn], [cost,lhs,rhs,rel,simi,simin,T.sum(out),cost,gradientsembeddings],updates=updates)
    #return theano.function([lrparams,lrembeddings,inpposr, inpposl, inpposo, inpposrn,maskE], [cost,lhs,rhs,rel,simi,simin,T.sum(out),cost,gradientsembeddings,(1+T.cast(T.sum(out),dtype=theano.config.floatX))],updates=updates)
    #return theano.function([lrparams,lrembeddings,inpposr, inpposl, inpposo, inpposrn], [cost], updates = updates)

