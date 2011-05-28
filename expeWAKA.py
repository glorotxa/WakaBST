import scipy.sparse
import cPickle
import os
import sys
from model import *
import time

def createrandommat(shape):
    randommat = scipy.sparse.lil_matrix((shape[0],shape[1]),dtype=theano.config.floatX)
    idxr = numpy.asarray(numpy.random.permutation(shape[1]),dtype='int32')
    idx = 0
    for i in idxr:
        if idx == shape[0]:
           idx=0
        randommat[idx,i]=1
        idx+=1
    return randommat.tocsr()

def expeWAKA(state,channel):
    state.savepath = channel.remote_path+'/' if hasattr(channel,'remote_path') else channel.path+'/'
    datpath = '/mnt/scratch/bengio/glorotxa/data/exp/WakaBST4/'

    state.listconcept = [['__brain_NN_1'],  ['__france_NN_1'], ['__auto_NN_1'],['__cat_NN_1'],['__monkey_NN_1'],['__u.s._NN_1','__army_NN_1']]
    state.listrel = [['_has_part'],['_part_of'],['__eat_VB_1'],['__drive_VB_1'],['__defend_VB_1'],['__attack_VB_1']]  
 
    dictparam = {}
    dictparam.update({ 'operator':state.operator})
    dictparam.update({ 'updateWN' :  state.updateWN})
    dictparam.update({ 'updateWNl' :  state.updateWNl})
    dictparam.update({ 'updateWNsl' :  state.updateWNsl})
    dictparam.update({ 'updateCN' :  state.updateCN})
    dictparam.update({ 'updateWK' :  state.updateWK})
    dictparam.update({ 'updateWKs' :  state.updateWKs})
    dictparam.update({ 'updateXWN' :  state.updateXWN}) 
    dictparam.update({ 'ndim' : state.ndim})
    dictparam.update({ 'nbbatches' : state.nbatches})
    dictparam.update({ 'lrparam' : state.lrparam})
    dictparam.update({ 'lremb' : state.lremb})
    dictparam.update({ 'nbtest' : state.nbtest})
    dictparam.update({ 'testall' : state.testall})
    dictparam.update({ 'savepath' : state.savepath})
    dictparam.update({ 'simfnstr' : state.simfnstr})
    dictparam.update({ 'listconcept' : state.listconcept})
    dictparam.update({ 'listrel' : state.listrel})
    dictparam.update({ 'nbrank' : state.nbrank})
    dictparam.update({ 'loadmodel' : state.loadmodel})
    dictparam.update({ 'begindeclr' : state.begindeclr})
    dictparam.update({ 'ratdeclr' : state.ratdeclr})
    dictparam.update({ 'totbatch' : state.totbatch})
    dictparam.update({ 'margewn' : state.margewn})
    dictparam.update({ 'margewnl' : state.margewnl})
    dictparam.update({ 'margewnsl' : state.margewnsl})
    dictparam.update({ 'margecn' : state.margecn})
    dictparam.update({ 'margewk' : state.margewk})
    dictparam.update({ 'margewks' : state.margewks})
    dictparam.update({ 'margexwn' : state.margexwn})
    dictparam.update({ 'relb' : state.relb})
    dictparam.update({ 'random' : state.random})
    
    
    print >> sys.stderr, 'operator : ', state.operator
    print >> sys.stderr, 'updateWN : ', state.updateWN
    print >> sys.stderr, 'updateWNl : ', state.updateWNl
    print >> sys.stderr, 'updateWNsl : ', state.updateWNsl
    print >> sys.stderr, 'updateCN : ', state.updateCN
    print >> sys.stderr, 'updateWK : ', state.updateWK
    print >> sys.stderr, 'updateWKs : ', state.updateWKs
    print >> sys.stderr, 'updateXWN : ', state.updateXWN
    print >> sys.stderr, 'ndim : ',  state.ndim
    print >> sys.stderr, 'nbbatches : ', state.nbatches
    print >> sys.stderr, 'lrparam : ', state.lrparam
    print >> sys.stderr, 'lremb : ', state.lremb
    print >> sys.stderr, 'nbtest : ', state.nbtest
    print >> sys.stderr, 'testall : ', state.testall
    print >> sys.stderr, 'savepath : ', state.savepath
    print >> sys.stderr, 'simfnstr : ', state.simfnstr
    print >> sys.stderr, 'listconcept : ', state.listconcept
    print >> sys.stderr, 'listrel : ', state.listrel
    print >> sys.stderr, 'nbrank : ', state.nbrank
    print >> sys.stderr, 'loadmodel : ', state.loadmodel
    print >> sys.stderr, 'begindeclr: ', state.begindeclr
    print >> sys.stderr, 'ratdeclr: ', state.ratdeclr
    print >> sys.stderr, 'totbatch: ', state.totbatch
    print >> sys.stderr, 'margewn: ', state.margewn
    print >> sys.stderr, 'margewnl: ', state.margewnl
    print >> sys.stderr, 'margewnsl: ', state.margewnsl
    print >> sys.stderr, 'margecn: ', state.margecn
    print >> sys.stderr, 'margewk: ', state.margewk
    print >> sys.stderr, 'margewks: ', state.margewks
    print >> sys.stderr, 'margexwn: ', state.margexwn
    print >> sys.stderr, 'relb: ', state.relb
    print >> sys.stderr, 'random: ', state.random

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

    print '####### WORDNET'
    # train set
    WNtrainl = (cPickle.load(open(datpath+'WordNet3.0-train-lhs.pkl','r'))).tocsr()
    WNtrainr = (cPickle.load(open(datpath+'WordNet3.0-train-rhs.pkl','r'))).tocsr()
    WNtraino = (cPickle.load(open(datpath+'WordNet3.0-train-rel.pkl','r'))).tocsr()

    if not state.random:
        WNtrainln = (cPickle.load(open(datpath+'WordNet3.0-train-lhs.pkl','r'))).tocsr()
        WNtrainrn = (cPickle.load(open(datpath+'WordNet3.0-train-rhs.pkl','r'))).tocsr()
        WNtrainon = (cPickle.load(open(datpath+'WordNet3.0-train-rel.pkl','r'))).tocsr()
    else:
        WNtrainln = createrandommat(WNtrainl.shape)
        WNtrainrn = createrandommat(WNtrainl.shape)
        WNtrainon = (cPickle.load(open(datpath+'WordNet3.0-train-rel.pkl','r'))).tocsr()
        #WNtrainon = createrandommat(WNtrainl.shape)

    numpy.random.seed(111)
    order = numpy.random.permutation(WNtrainl.shape[1])
    WNtrainl = WNtrainl[:,order]
    WNtrainr = WNtrainr[:,order]
    WNtraino = WNtraino[:,order]
    WNtrainln = WNtrainln[:,numpy.random.permutation(WNtrainln.shape[1])]
    WNtrainrn = WNtrainrn[:,numpy.random.permutation(WNtrainln.shape[1])]
    WNtrainon = WNtrainon[:,numpy.random.permutation(WNtrainln.shape[1])]

    # valid set
    WNvall = (cPickle.load(open(datpath+'WordNet3.0-val-lhs.pkl','r'))).tocsr()
    WNvalr = (cPickle.load(open(datpath+'WordNet3.0-val-rhs.pkl','r'))).tocsr()
    WNvalo = (cPickle.load(open(datpath+'WordNet3.0-val-rel.pkl','r'))).tocsr()
    numpy.random.seed(222)
    order = numpy.random.permutation(WNvall.shape[1])
    WNvall = WNvall[:,order]
    WNvalr = WNvalr[:,order]
    WNvalo = WNvalo[:,order]

    # test set
    WNtestl = (cPickle.load(open(datpath+'WordNet3.0-test-lhs.pkl','r'))).tocsr()
    WNtestr = (cPickle.load(open(datpath+'WordNet3.0-test-rhs.pkl','r'))).tocsr()
    WNtesto = (cPickle.load(open(datpath+'WordNet3.0-test-rel.pkl','r'))).tocsr()
    numpy.random.seed(333)
    order = numpy.random.permutation(WNtestl.shape[1])
    WNtestl = WNtestl[:,order]
    WNtestr = WNtestr[:,order]
    WNtesto = WNtesto[:,order]

 
    print '####### WORDNET LEMME'
    # train set
    WNltrainl = (cPickle.load(open(datpath+'WordNet3.0-lemme-train-lhs.pkl','r'))).tocsr()
    WNltrainr = (cPickle.load(open(datpath+'WordNet3.0-lemme-train-rhs.pkl','r'))).tocsr()
    WNltraino = (cPickle.load(open(datpath+'WordNet3.0-lemme-train-rel.pkl','r'))).tocsr()

    if not state.random:
        WNltrainln = (cPickle.load(open(datpath+'WordNet3.0-lemme-train-lhs.pkl','r'))).tocsr()
        WNltrainrn = (cPickle.load(open(datpath+'WordNet3.0-lemme-train-rhs.pkl','r'))).tocsr()
        WNltrainon = (cPickle.load(open(datpath+'WordNet3.0-lemme-train-rel.pkl','r'))).tocsr()
    else:
        WNltrainln = createrandommat(WNltrainl.shape)
        WNltrainrn = createrandommat(WNltrainl.shape)
        WNltrainon = (cPickle.load(open(datpath+'WordNet3.0-lemme-train-rel.pkl','r'))).tocsr()
        #WNltrainon = createrandommat(WNltrainl.shape)

    numpy.random.seed(222)
    order = numpy.random.permutation(WNltrainl.shape[1])
    WNltrainl = WNltrainl[:,order]
    WNltrainr = WNltrainr[:,order]
    WNltraino = WNltraino[:,order]
    WNltrainln = WNltrainln[:,numpy.random.permutation(WNltrainln.shape[1])]
    WNltrainrn = WNltrainrn[:,numpy.random.permutation(WNltrainln.shape[1])]
    WNltrainon = WNltrainon[:,numpy.random.permutation(WNltrainln.shape[1])]

    print '####### WORDNET syle'
    # train set
    WNsltrainl = (cPickle.load(open(datpath+'WordNet3.0-syle-train-lhs.pkl','r'))).tocsr()
    WNsltrainr = (cPickle.load(open(datpath+'WordNet3.0-syle-train-rhs.pkl','r'))).tocsr()
    WNsltraino = (cPickle.load(open(datpath+'WordNet3.0-syle-train-rel.pkl','r'))).tocsr()

    if not state.random:
        WNsltrainln = (cPickle.load(open(datpath+'WordNet3.0-syle-train-lhs.pkl','r'))).tocsr()
        WNsltrainrn = (cPickle.load(open(datpath+'WordNet3.0-syle-train-rhs.pkl','r'))).tocsr()
        WNsltrainon = (cPickle.load(open(datpath+'WordNet3.0-syle-train-rel.pkl','r'))).tocsr()
    else:
        WNsltrainln = createrandommat(WNsltrainl.shape)
        WNsltrainrn = createrandommat(WNsltrainl.shape)
        WNsltrainon = (cPickle.load(open(datpath+'WordNet3.0-syle-train-rel.pkl','r'))).tocsr()
        #WNsltrainon = createrandommat(WNsltrainl.shape)

    numpy.random.seed(333)
    order = numpy.random.permutation(WNsltrainl.shape[1])
    WNsltrainl = WNsltrainl[:,order]
    WNsltrainr = WNsltrainr[:,order]
    WNsltraino = WNsltraino[:,order]
    WNsltrainln = WNsltrainln[:,numpy.random.permutation(WNsltrainln.shape[1])]
    WNsltrainrn = WNsltrainrn[:,numpy.random.permutation(WNsltrainln.shape[1])]
    WNsltrainon = WNsltrainon[:,numpy.random.permutation(WNsltrainln.shape[1])]    

    print '####### ConceptNet' 
    CNtrainl = (cPickle.load(open(datpath+'ConceptNet-lhs.pkl','r'))).tocsr()
    CNtrainr = (cPickle.load(open(datpath+'ConceptNet-rhs.pkl','r'))).tocsr()
    CNtraino = (cPickle.load(open(datpath+'ConceptNet-rel.pkl','r'))).tocsr()
    
    if not state.random:
        CNtrainln = (cPickle.load(open(datpath+'ConceptNet-lhs.pkl','r'))).tocsr()
        CNtrainrn = (cPickle.load(open(datpath+'ConceptNet-rhs.pkl','r'))).tocsr()
        CNtrainon = (cPickle.load(open(datpath+'ConceptNet-rel.pkl','r'))).tocsr()
    else:
        CNtrainln = createrandommat(CNtrainl.shape)
        CNtrainrn = createrandommat(CNtrainl.shape)
        CNtrainon = (cPickle.load(open(datpath+'ConceptNet-rel.pkl','r'))).tocsr()
    
    numpy.random.seed(444)
    order = numpy.random.permutation(CNtrainl.shape[1])
    CNtrainl = CNtrainl[:,order]
    CNtrainr = CNtrainr[:,order]
    CNtraino = CNtraino[:,order]
    CNtrainln = CNtrainln[:,numpy.random.permutation(CNtrainln.shape[1])]
    CNtrainrn = CNtrainrn[:,numpy.random.permutation(CNtrainln.shape[1])]
    CNtrainon = CNtrainon[:,numpy.random.permutation(CNtrainln.shape[1])]
    
    print '####### Wikisample'
    WKtrainl = (cPickle.load(open(datpath+'Wikisample-lhs.pkl','r'))).tocsr()
    WKtrainr = (cPickle.load(open(datpath+'Wikisample-rhs.pkl','r'))).tocsr()
    WKtraino = (cPickle.load(open(datpath+'Wikisample-rel.pkl','r'))).tocsr()

    if not state.random:
        WKtrainln = (cPickle.load(open(datpath+'Wikisample-lhs.pkl','r'))).tocsr()
        WKtrainrn = (cPickle.load(open(datpath+'Wikisample-rhs.pkl','r'))).tocsr()
        WKtrainon = (cPickle.load(open(datpath+'Wikisample-rel.pkl','r'))).tocsr()
    else:
        WKtrainln = createrandommat((WNtrainl.shape[0],WNtrainl.shape[1]/state.nbatches*20+11000))
        WKtrainrn = createrandommat((WNtrainl.shape[0],WNtrainl.shape[1]/state.nbatches*20+11000))
        WKtrainon = createrandommat((WNtrainl.shape[0],WNtrainl.shape[1]/state.nbatches*20+11000))

    numpy.random.seed(555)
    order = numpy.random.permutation(WKtrainl.shape[1])
    WKtrainl = WKtrainl[:,order]
    WKtrainr = WKtrainr[:,order]
    WKtraino = WKtraino[:,order]
    WKtrainln = WKtrainln[:,numpy.random.permutation(WKtrainln.shape[1])]
    WKtrainrn = WKtrainrn[:,numpy.random.permutation(WKtrainln.shape[1])]
    WKtrainon = WKtrainon[:,numpy.random.permutation(WKtrainln.shape[1])]

    WKtrainvall = WKtrainl[:,-10000:]
    WKtrainvalr = WKtrainr[:,-10000:]
    WKtrainvalo = WKtraino[:,-10000:]
    WKtrainvalln = WKtrainln[:,-10000:]
    WKtrainvalrn = WKtrainrn[:,-10000:]
    WKtrainvalon = WKtrainon[:,-10000:]

    WKtrainl = WKtrainl[:,:-10000]
    WKtrainr = WKtrainr[:,:-10000]
    WKtraino = WKtraino[:,:-10000]
    WKtrainln = WKtrainln[:,:-10000]
    WKtrainrn = WKtrainrn[:,:-10000]
    WKtrainon = WKtrainon[:,:-10000]

    print '####### Wikisuper'
    WKstrainl = (cPickle.load(open(datpath+'Wikisuper-lhs.pkl','r'))).tocsr()
    WKstrainr = (cPickle.load(open(datpath+'Wikisuper-rhs.pkl','r'))).tocsr()
    WKstraino = (cPickle.load(open(datpath+'Wikisuper-rel.pkl','r'))).tocsr()

    WKstrainln = (cPickle.load(open(datpath+'Wikisuper-lhsn.pkl','r'))).tocsr()
    WKstrainrn = (cPickle.load(open(datpath+'Wikisuper-rhsn.pkl','r'))).tocsr()
    WKstrainon = (cPickle.load(open(datpath+'Wikisuper-reln.pkl','r'))).tocsr()

    WKstrainvall = WKstrainl[:,-10000:]
    WKstrainvalr = WKstrainr[:,-10000:]
    WKstrainvalo = WKstraino[:,-10000:]
    WKstrainvalln = WKstrainln[:,-10000:]
    WKstrainvalrn = WKstrainrn[:,-10000:]
    WKstrainvalon = WKstrainon[:,-10000:]

    WKstrainl = WKstrainl[:,:-10000]
    WKstrainr = WKstrainr[:,:-10000]
    WKstraino = WKstraino[:,:-10000]
    WKstrainln = WKstrainln[:,:-10000]
    WKstrainrn = WKstrainrn[:,:-10000]
    WKstrainon = WKstrainon[:,:-10000]

    numpy.random.seed(666)
    order = numpy.random.permutation(WKstrainl.shape[1])
    WKstrainl = WKstrainl[:,order]
    WKstrainr = WKstrainr[:,order]
    WKstraino = WKstraino[:,order]
    WKstrainln = WKstrainln[:,order]
    WKstrainrn = WKstrainrn[:,order]
    WKstrainon = WKstrainon[:,order]

    print '####### XWN'
    XWNtrainl = (cPickle.load(open(datpath+'XWN-lhs.pkl','r'))).tocsr()
    XWNtrainr = (cPickle.load(open(datpath+'XWN-rhs.pkl','r'))).tocsr()
    XWNtraino = (cPickle.load(open(datpath+'XWN-rel.pkl','r'))).tocsr()

    XWNtrainln = (cPickle.load(open(datpath+'XWN-lhsn.pkl','r'))).tocsr()
    XWNtrainrn = (cPickle.load(open(datpath+'XWN-rhsn.pkl','r'))).tocsr()
    XWNtrainon = (cPickle.load(open(datpath+'XWN-reln.pkl','r'))).tocsr()

    XWNtrainvall = XWNtrainl[:,-10000:]
    XWNtrainvalr = XWNtrainr[:,-10000:]
    XWNtrainvalo = XWNtraino[:,-10000:]
    XWNtrainvalln = XWNtrainln[:,-10000:]
    XWNtrainvalrn = XWNtrainrn[:,-10000:]
    XWNtrainvalon = XWNtrainon[:,-10000:]

    XWNtrainl = XWNtrainl[:,:-10000]
    XWNtrainr = XWNtrainr[:,:-10000]
    XWNtraino = XWNtraino[:,:-10000]
    XWNtrainln = XWNtrainln[:,:-10000]
    XWNtrainrn = XWNtrainrn[:,:-10000]
    XWNtrainon = XWNtrainon[:,:-10000]

    numpy.random.seed(777)
    order = numpy.random.permutation(XWNtrainl.shape[1])
    XWNtrainl = XWNtrainl[:,order]
    XWNtrainr = XWNtrainr[:,order]
    XWNtraino = XWNtraino[:,order]
    XWNtrainln = XWNtrainln[:,order]
    XWNtrainrn = XWNtrainrn[:,order]
    XWNtrainon = XWNtrainon[:,order]

    # ------------------
    rows,cols = WNtestl.nonzero()
    idxtl = rows[numpy.argsort(cols)]
    rows,cols = WNtestr.nonzero()
    idxtr = rows[numpy.argsort(cols)]
    rows,cols = WNtesto.nonzero()
    idxto = rows[numpy.argsort(cols)]

    rows,cols = WNvall.nonzero()
    idxvl = rows[numpy.argsort(cols)]
    rows,cols = WNvalr.nonzero()
    idxvr = rows[numpy.argsort(cols)]
    rows,cols = WNvalo.nonzero()
    idxvo = rows[numpy.argsort(cols)]

    if not state.loadmodel:
        # operators
        if  state.operator == 'Id':
            leftop = Id()
            rightop = Id()
        elif  state.operator == 'linear':
            leftop = Layercomb(numpy.random, 'lin', state.ndim, state.ndim, state.ndim)
            rightop = Layercomb(numpy.random, 'lin', state.ndim, state.ndim, state.ndim)
        elif  state.operator == 'mlp':
            leftop = MLP(numpy.random, 'sigm', state.ndim, state.ndim, (3*state.ndim)/2, state.ndim)
            rightop = MLP(numpy.random, 'sigm', state.ndim, state.ndim, (3*state.ndim)/2, state.ndim)
        elif  state.operator == 'quad':
            leftop = Quadlayer(numpy.random, state.ndim, state.ndim, (3*state.ndim)/2, state.ndim)
            rightop = Quadlayer(numpy.random, state.ndim, state.ndim, (3*state.ndim)/2, state.ndim)
        if state.simfnstr == 'MLP':
            MLPout = MLP(numpy.random, 'sigm', state.ndim, state.ndim, state.ndim, 1)
        # embeddings
        embeddings = Embedd(numpy.random,numpy.max(lemme2idx.values())+1,state.ndim)
    else:
        f = open(state.loadmodel)
        embeddings = cPickle.load(f)
        leftop = cPickle.load(f)
        rightop = cPickle.load(f)
        if state.simfnstr == 'MLP':
           MLPout = cPickle.load(f)
        f.close()
        dictparam = cPickle.load(open(state.loadmodel[:-4]+'dict.pkl'))
    
    if state.simfnstr == 'MLP':
        simfn = MLPout
    else:
        simfn = eval(state.simfnstr+'sim')


    # train function
    ftwn = TrainFunction(simfn,embeddings,leftop,rightop, marge = state.margewn, relb = state.relb)
    ftwnl = TrainFunction(simfn,embeddings,leftop,rightop, marge = state.margewnl, relb = state.relb)
    ftwnsl = TrainFunction(simfn,embeddings,leftop,rightop, marge = state.margewnsl, relb = state.relb)
    ftcn = TrainFunction(simfn,embeddings,leftop,rightop, marge = state.margecn, relb = state.relb)
    ftwk = TrainFunction(simfn,embeddings,leftop,rightop, marge = state.margewk, relb = state.relb)
    ftwks = TrainFunction(simfn,embeddings,leftop,rightop, marge = state.margewks, relb = state.relb)
    ftxwn = TrainFunction(simfn,embeddings,leftop,rightop, marge = state.margexwn, relb = state.relb)
    vt = BatchValidFunction(simfn,embeddings,leftop,rightop)

    # simi function
    # for the right Word Net
    sl = SimilarityFunctionleft(simfn,embeddings,leftop,rightop,subtensorspec = numpy.max(synset2idx.values())+1)
    sr = SimilarityFunctionright(simfn,embeddings,leftop,rightop,subtensorspec = numpy.max(synset2idx.values())+1)
    srl = SimilarityFunctionrightl(simfn,embeddings,leftop,rightop)
    sll = SimilarityFunctionleftl(simfn,embeddings,leftop,rightop)
    sol = SimilarityFunctionrell(simfn,embeddings,leftop,rightop)
    leftopid = Id()
    rightopid = Id()
    Esim = SimilarityFunctionrightl(L2sim,embeddings,leftopid,rightopid)

    if 'epochl' in dictparam.keys():
        ct = dictparam['epochl'][-1]
    else:
        ct = 0
        dictparam['epochl'] = []
        dictparam['lrembl'] = []
        dictparam['lrparaml'] = []

        dictparam['WNallmean'] = []
        dictparam['WNallstd']  = []
        dictparam['WNleftmean'] = []
        dictparam['WNleftstd'] = []
        dictparam['WNrightmean'] = []
        dictparam['WNrightstd'] = []
        dictparam['WNrelamean'] = []
        dictparam['WNrelastd'] = []
        dictparam['WNallbmean'] = []
        dictparam['WNallbstd']  = []
        dictparam['WNleftbmean'] = []
        dictparam['WNleftbstd'] = []
        dictparam['WNrightbmean'] = []
        dictparam['WNrightbstd'] = []
        dictparam['WNrelabmean'] = []
        dictparam['WNrelabstd'] = []

        dictparam['WNlallmean'] = []
        dictparam['WNlallstd']  = []
        dictparam['WNlleftmean'] = []
        dictparam['WNlleftstd'] = []
        dictparam['WNlrightmean'] = []
        dictparam['WNlrightstd'] = []
        dictparam['WNlrelamean'] = []
        dictparam['WNlrelastd'] = []
        dictparam['WNlallbmean'] = []
        dictparam['WNlallbstd']  = []
        dictparam['WNlleftbmean'] = []
        dictparam['WNlleftbstd'] = []
        dictparam['WNlrightbmean'] = []
        dictparam['WNlrightbstd'] = []
        dictparam['WNlrelabmean'] = []
        dictparam['WNlrelabstd'] = []

        dictparam['WNslallmean'] = []
        dictparam['WNslallstd']  = []
        dictparam['WNslleftmean'] = []
        dictparam['WNslleftstd'] = []
        dictparam['WNslrightmean'] = []
        dictparam['WNslrightstd'] = []
        dictparam['WNslrelamean'] = []
        dictparam['WNslrelastd'] = []
        dictparam['WNslallbmean'] = []
        dictparam['WNslallbstd']  = []
        dictparam['WNslleftbmean'] = []
        dictparam['WNslleftbstd'] = []
        dictparam['WNslrightbmean'] = []
        dictparam['WNslrightbstd'] = []
        dictparam['WNslrelabmean'] = []
        dictparam['WNslrelabstd'] = []

        dictparam['CNallmean'] = []
        dictparam['CNallstd']  = []
        dictparam['CNleftmean'] = []
        dictparam['CNleftstd'] = []
        dictparam['CNrightmean'] = []
        dictparam['CNrightstd'] = []
        dictparam['CNrelamean'] = []
        dictparam['CNrelastd'] = []
        dictparam['CNallbmean'] = []
        dictparam['CNallbstd']  = []
        dictparam['CNleftbmean'] = []
        dictparam['CNleftbstd'] = []
        dictparam['CNrightbmean'] = []
        dictparam['CNrightbstd'] = []
        dictparam['CNrelabmean'] = []
        dictparam['CNrelabstd'] = []

        dictparam['WKallmean'] = []
        dictparam['WKallstd']  = []
        dictparam['WKleftmean'] = []
        dictparam['WKleftstd'] = []
        dictparam['WKrightmean'] = []
        dictparam['WKrightstd'] = []
        dictparam['WKrelamean'] = []
        dictparam['WKrelastd'] = []
        dictparam['WKallbmean'] = []
        dictparam['WKallbstd']  = []
        dictparam['WKleftbmean'] = []
        dictparam['WKleftbstd'] = []
        dictparam['WKrightbmean'] = []
        dictparam['WKrightbstd'] = []
        dictparam['WKrelabmean'] = []
        dictparam['WKrelabstd'] = []

        dictparam['WKsallmean'] = []
        dictparam['WKsallstd']  = []
        dictparam['WKsleftmean'] = []
        dictparam['WKsleftstd'] = []
        dictparam['WKsrightmean'] = []
        dictparam['WKsrightstd'] = []
        dictparam['WKsrelamean'] = []
        dictparam['WKsrelastd'] = []
        dictparam['WKsallbmean'] = []
        dictparam['WKsallbstd']  = []
        dictparam['WKsleftbmean'] = []
        dictparam['WKsleftbstd'] = []
        dictparam['WKsrightbmean'] = []
        dictparam['WKsrightbstd'] = []
        dictparam['WKsrelabmean'] = []
        dictparam['WKsrelabstd'] = []

        dictparam['XWNallmean'] = []
        dictparam['XWNallstd']  = []
        dictparam['XWNleftmean'] = []
        dictparam['XWNleftstd'] = []
        dictparam['XWNrightmean'] = []
        dictparam['XWNrightstd'] = []
        dictparam['XWNrelamean'] = []
        dictparam['XWNrelastd'] = []
        dictparam['XWNallbmean'] = []
        dictparam['XWNallbstd']  = []
        dictparam['XWNleftbmean'] = []
        dictparam['XWNleftbstd'] = []
        dictparam['XWNrightbmean'] = []
        dictparam['XWNrightbstd'] = []
        dictparam['XWNrelabmean'] = []
        dictparam['XWNrelabstd'] = []
        
        dictparam['WNval'] = []
        dictparam['WNtes'] = []
        dictparam['WKval'] = []
        dictparam['WKvalb'] = []
        dictparam['WKvalm'] = []
        dictparam['WKsval'] = []
        dictparam['WKsvalb'] = []
        dictparam['WKsvalm'] = []
        dictparam['XWNval'] = []
        dictparam['XWNvalb'] = []
        dictparam['XWNvalm'] = []


    WNleft = []
    WNright = []
    WNrela = []
    WNleftb = []
    WNrightb = []
    WNrelab = []

    WNlleft = []
    WNlright = []
    WNlrela = []
    WNlleftb = []
    WNlrightb = []
    WNlrelab = []

    WNslleft = []
    WNslright = []
    WNslrela = []
    WNslleftb = []
    WNslrightb = []
    WNslrelab = []

    CNleft = []
    CNright = []
    CNrela = []
    CNleftb = []
    CNrightb = []
    CNrelab = []

    WKleft = []
    WKright = []
    WKrela = []
    WKleftb = []
    WKrightb = []
    WKrelab = []

    WKsleft = []
    WKsright = []
    WKsrela = []
    WKsleftb = []
    WKsrightb = []
    WKsrelab = []

    XWNleft = []
    XWNright = []
    XWNrela = []
    XWNleftb = []
    XWNrightb = []
    XWNrelab = []


    state.bestWNval = -1
    state.bestWNtes = -1
    state.bestWKval = -1
    state.bestWKvalm = -1
    state.bestWKvalb = -1
    state.bestWKsval = -1
    state.bestWKsvalm = -1
    state.bestWKsvalb = -1
    state.bestXWNval = -1
    state.bestXWNvalm = -1
    state.bestXWNvalb = -1

    M = WNtrainl.shape[1]/state.nbatches
    WNlbatch = WNltrainl.shape[1] / M
    WNlbatchct=0
    WNslbatch = WNsltrainl.shape[1] / M
    WNslbatchct=0
    CNbatch = CNtrainl.shape[1] / M
    CNbatchct=0
    WKbatch = WKtrainl.shape[1] / M
    WKbatchct=0
    WKnegbatch = 20
    WKnegbatchct = 0
    WKsbatch = WKstrainl.shape[1] / M
    WKsbatchct=0
    XWNbatch = XWNtrainl.shape[1] / M
    XWNbatchct=0
    
    ref = time.time()
    print >> sys.stderr, "BEGIN TRAINING"
    for ccc in range(state.totbatch): 
        for i in range(state.nbatches):
            if state.updateWN:
                if ct > state.begindeclr:
                    resl = ftwn(state.lrparam/float((1+state.ratdeclr * (ct-state.begindeclr))*float(M)),state.lremb/float(1+state.ratdeclr * (ct-state.begindeclr)),WNtrainl[:,i*M:(i+1)*M],WNtrainr[:,i*M:(i+1)*M],WNtraino[:,i*M:(i+1)*M],WNtrainln[:,i*M:(i+1)*M],WNtrainrn[:,i*M:(i+1)*M],WNtrainon[:,i*M:(i+1)*M])
                else:
                    resl = ftwn(state.lrparam/float(M),state.lremb,WNtrainl[:,i*M:(i+1)*M],WNtrainr[:,i*M:(i+1)*M],WNtraino[:,i*M:(i+1)*M],WNtrainln[:,i*M:(i+1)*M],WNtrainrn[:,i*M:(i+1)*M],WNtrainon[:,i*M:(i+1)*M])
                WNleft += [resl[1]/float(M)]
                WNright += [resl[2]/float(M)]
                WNrela += [resl[3]/float(M)]
                WNleftb += [resl[5]/float(M)]
                WNrightb += [resl[6]/float(M)]
                WNrelab += [resl[7]/float(M)]
                embeddings.norma()
            
            if state.updateWNl:
                if WNlbatchct == WNlbatch:
                    WNltrainln = WNltrainln[:,numpy.random.permutation(WNltrainln.shape[1])]
                    WNltrainrn = WNltrainrn[:,numpy.random.permutation(WNltrainln.shape[1])]
                    WNltrainon = WNltrainon[:,numpy.random.permutation(WNltrainln.shape[1])]
                    neworder = numpy.random.permutation(WNltrainln.shape[1])
                    WNltrainl = WNltrainl[:,neworder]
                    WNltrainr = WNltrainr[:,neworder]
                    WNltraino = WNltraino[:,neworder]
                    WNlbatchct = 0
                if ct > state.begindeclr:
                    resl = ftwnl(state.lrparam/float((1+state.ratdeclr * (ct-state.begindeclr))*float(M)),state.lremb/float(1+state.ratdeclr * (ct-state.begindeclr)),WNltrainl[:,WNlbatchct*M:(WNlbatchct+1)*M],WNltrainr[:,WNlbatchct*M:(WNlbatchct+1)*M],WNltraino[:,WNlbatchct*M:(WNlbatchct+1)*M],WNltrainln[:,WNlbatchct*M:(WNlbatchct+1)*M],WNltrainrn[:,WNlbatchct*M:(WNlbatchct+1)*M],WNltrainon[:,WNlbatchct*M:(WNlbatchct+1)*M])
                else:
                    resl = ftwnl(state.lrparam/float(M),state.lremb,WNltrainl[:,WNlbatchct*M:(WNlbatchct+1)*M],WNltrainr[:,WNlbatchct*M:(WNlbatchct+1)*M],WNltraino[:,WNlbatchct*M:(WNlbatchct+1)*M],WNltrainln[:,WNlbatchct*M:(WNlbatchct+1)*M],WNltrainrn[:,WNlbatchct*M:(WNlbatchct+1)*M],WNltrainon[:,WNlbatchct*M:(WNlbatchct+1)*M])
                WNlleft += [resl[1]/float(M)]
                WNlright += [resl[2]/float(M)]
                WNlrela += [resl[3]/float(M)]
                WNlleftb += [resl[5]/float(M)]
                WNlrightb += [resl[6]/float(M)]
                WNlrelab += [resl[7]/float(M)]
                embeddings.norma()
                WNlbatchct += 1
            
            if state.updateWNsl:
                if WNslbatchct == WNslbatch:
                    WNsltrainln = WNsltrainln[:,numpy.random.permutation(WNsltrainln.shape[1])]
                    WNsltrainrn = WNsltrainrn[:,numpy.random.permutation(WNsltrainln.shape[1])]
                    WNsltrainon = WNsltrainon[:,numpy.random.permutation(WNsltrainln.shape[1])]
                    neworder = numpy.random.permutation(WNsltrainln.shape[1])
                    WNsltrainl = WNsltrainl[:,neworder]
                    WNsltrainr = WNsltrainr[:,neworder]
                    WNsltraino = WNsltraino[:,neworder]
                    WNslbatchct = 0
                if ct > state.begindeclr:
                    resl = ftwnsl(state.lrparam/float((1+state.ratdeclr * (ct-state.begindeclr))*float(M)),state.lremb/float(1+state.ratdeclr * (ct-state.begindeclr)),WNsltrainl[:,WNslbatchct*M:(WNslbatchct+1)*M],WNsltrainr[:,WNslbatchct*M:(WNslbatchct+1)*M],WNsltraino[:,WNslbatchct*M:(WNslbatchct+1)*M],WNsltrainln[:,WNslbatchct*M:(WNslbatchct+1)*M],WNsltrainrn[:,WNslbatchct*M:(WNslbatchct+1)*M],WNsltrainon[:,WNslbatchct*M:(WNslbatchct+1)*M])
                else:
                    resl = ftwnsl(state.lrparam/float(M),state.lremb,WNsltrainl[:,WNslbatchct*M:(WNslbatchct+1)*M],WNsltrainr[:,WNslbatchct*M:(WNslbatchct+1)*M],WNsltraino[:,WNslbatchct*M:(WNslbatchct+1)*M],WNsltrainln[:,WNslbatchct*M:(WNslbatchct+1)*M],WNsltrainrn[:,WNslbatchct*M:(WNslbatchct+1)*M],WNsltrainon[:,WNslbatchct*M:(WNslbatchct+1)*M])
                WNslleft += [resl[1]/float(M)]
                WNslright += [resl[2]/float(M)]
                WNslrela += [resl[3]/float(M)]
                WNslleftb += [resl[5]/float(M)]
                WNslrightb += [resl[6]/float(M)]
                WNslrelab += [resl[7]/float(M)]
                embeddings.norma()
                WNslbatchct += 1

            if state.updateCN:
                if CNbatchct == CNbatch:
                    CNtrainln = CNtrainln[:,numpy.random.permutation(CNtrainln.shape[1])]
                    CNtrainrn = CNtrainrn[:,numpy.random.permutation(CNtrainln.shape[1])]
                    CNtrainon = CNtrainon[:,numpy.random.permutation(CNtrainln.shape[1])]
                    neworder = numpy.random.permutation(CNtrainln.shape[1])
                    CNtrainl = CNtrainl[:,neworder]
                    CNtrainr = CNtrainr[:,neworder]
                    CNtraino = CNtraino[:,neworder]
                    CNbatchct = 0
                if ct > state.begindeclr:
                    resl = ftcn(state.lrparam/float((1+state.ratdeclr * (ct-state.begindeclr))*float(M)),state.lremb/float(1+state.ratdeclr * (ct-state.begindeclr)),CNtrainl[:,CNbatchct*M:(CNbatchct+1)*M],CNtrainr[:,CNbatchct*M:(CNbatchct+1)*M],CNtraino[:,CNbatchct*M:(CNbatchct+1)*M],CNtrainln[:,CNbatchct*M:(CNbatchct+1)*M],CNtrainrn[:,CNbatchct*M:(CNbatchct+1)*M],CNtrainon[:,CNbatchct*M:(CNbatchct+1)*M])
                else:
                    resl = ftcn(state.lrparam/float(M),state.lremb,CNtrainl[:,CNbatchct*M:(CNbatchct+1)*M],CNtrainr[:,CNbatchct*M:(CNbatchct+1)*M],CNtraino[:,CNbatchct*M:(CNbatchct+1)*M],CNtrainln[:,CNbatchct*M:(CNbatchct+1)*M],CNtrainrn[:,CNbatchct*M:(CNbatchct+1)*M],CNtrainon[:,CNbatchct*M:(CNbatchct+1)*M])
                CNleft += [resl[1]/float(M)]
                CNright += [resl[2]/float(M)]
                CNrela += [resl[3]/float(M)]
                CNleftb += [resl[5]/float(M)]
                CNrightb += [resl[6]/float(M)]
                CNrelab += [resl[7]/float(M)]
                embeddings.norma()
                CNbatchct += 1

            if state.updateWK:
                if WKnegbatchct == WKnegbatch:
                    WKtrainln = WKtrainln[:,numpy.random.permutation(WKtrainln.shape[1])]
                    WKtrainrn = WKtrainrn[:,numpy.random.permutation(WKtrainln.shape[1])]
                    WKtrainon = WKtrainon[:,numpy.random.permutation(WKtrainln.shape[1])]
                    WKnegbatchct = 0
                if WKbatchct == WKbatch:
                    neworder = numpy.random.permutation(WKtrainl.shape[1])
                    WKtrainl = WKtrainl[:,neworder]
                    WKtrainr = WKtrainr[:,neworder]
                    WKtraino = WKtraino[:,neworder]
                    WKbatchct = 0
                if ct > state.begindeclr:
                    resl = ftwk(state.lrparam/float((1+state.ratdeclr * (ct-state.begindeclr))*float(M)),state.lremb/float(1+state.ratdeclr * (ct-state.begindeclr)),WKtrainl[:,WKbatchct*M:(WKbatchct+1)*M],WKtrainr[:,WKbatchct*M:(WKbatchct+1)*M],WKtraino[:,WKbatchct*M:(WKbatchct+1)*M],WKtrainln[:,WKnegbatchct*M:(WKnegbatchct+1)*M],WKtrainrn[:,WKnegbatchct*M:(WKnegbatchct+1)*M],WKtrainon[:,WKnegbatchct*M:(WKnegbatchct+1)*M])
                else:
                    resl = ftwk(state.lrparam/float(M),state.lremb,WKtrainl[:,WKbatchct*M:(WKbatchct+1)*M],WKtrainr[:,WKbatchct*M:(WKbatchct+1)*M],WKtraino[:,WKbatchct*M:(WKbatchct+1)*M],WKtrainln[:,WKnegbatchct*M:(WKnegbatchct+1)*M],WKtrainrn[:,WKnegbatchct*M:(WKnegbatchct+1)*M],WKtrainon[:,WKnegbatchct*M:(WKnegbatchct+1)*M])
                WKleft += [resl[1]/float(M)]
                WKright += [resl[2]/float(M)]
                WKrela += [resl[3]/float(M)]
                WKleftb += [resl[5]/float(M)]
                WKrightb += [resl[6]/float(M)]
                WKrelab += [resl[7]/float(M)]
                embeddings.norma()
                WKbatchct += 1
                WKnegbatch += 1

            if state.updateWKs:
                if WKsbatchct == WKsbatch:
                    neworder = numpy.random.permutation(WKstrainln.shape[1])
                    WKstrainln = WKstrainln[:,neworder]
                    WKstrainrn = WKstrainrn[:,neworder]
                    WKstrainon = WKstrainon[:,neworder]
                    WKstrainl = WKstrainl[:,neworder]
                    WKstrainr = WKstrainr[:,neworder]
                    WKstraino = WKstraino[:,neworder]
                    WKsbatchct = 0
                if ct > state.begindeclr:
                    resl = ftwks(state.lrparam/float((1+state.ratdeclr * (ct-state.begindeclr))*float(M)),state.lremb/float(1+state.ratdeclr * (ct-state.begindeclr)),WKstrainl[:,WKsbatchct*M:(WKsbatchct+1)*M],WKstrainr[:,WKsbatchct*M:(WKsbatchct+1)*M],WKstraino[:,WKsbatchct*M:(WKsbatchct+1)*M],WKstrainln[:,WKsbatchct*M:(WKsbatchct+1)*M],WKstrainrn[:,WKsbatchct*M:(WKsbatchct+1)*M],WKstrainon[:,WKsbatchct*M:(WKsbatchct+1)*M])
                else:
                    resl = ftwks(state.lrparam/float(M),state.lremb,WKstrainl[:,WKsbatchct*M:(WKsbatchct+1)*M],WKstrainr[:,WKsbatchct*M:(WKsbatchct+1)*M],WKstraino[:,WKsbatchct*M:(WKsbatchct+1)*M],WKstrainln[:,WKsbatchct*M:(WKsbatchct+1)*M],WKstrainrn[:,WKsbatchct*M:(WKsbatchct+1)*M],WKstrainon[:,WKsbatchct*M:(WKsbatchct+1)*M])
                WKsleft += [resl[1]/float(M)]
                WKsright += [resl[2]/float(M)]
                WKsrela += [resl[3]/float(M)]
                WKsleftb += [resl[5]/float(M)]
                WKsrightb += [resl[6]/float(M)]
                WKsrelab += [resl[7]/float(M)]
                embeddings.norma()
                WKsbatchct += 1

            if state.updateXWN:
                if XWNbatchct == XWNbatch:
                    neworder = numpy.random.permutation(XWNtrainln.shape[1])
                    XWNtrainln = XWNtrainln[:,neworder]
                    XWNtrainrn = XWNtrainrn[:,neworder]
                    XWNtrainon = XWNtrainon[:,neworder]
                    XWNtrainl = XWNtrainl[:,neworder]
                    XWNtrainr = XWNtrainr[:,neworder]
                    XWNtraino = XWNtraino[:,neworder]
                    XWNbatchct = 0
                if ct > state.begindeclr:
                    resl = ftxwn(state.lrparam/float((1+state.ratdeclr * (ct-state.begindeclr))*float(M)),state.lremb/float(1+state.ratdeclr * (ct-state.begindeclr)),XWNtrainl[:,XWNbatchct*M:(XWNbatchct+1)*M],XWNtrainr[:,XWNbatchct*M:(XWNbatchct+1)*M],XWNtraino[:,XWNbatchct*M:(XWNbatchct+1)*M],XWNtrainln[:,XWNbatchct*M:(XWNbatchct+1)*M],XWNtrainrn[:,XWNbatchct*M:(XWNbatchct+1)*M],XWNtrainon[:,XWNbatchct*M:(XWNbatchct+1)*M])
                else:
                    resl = ftxwn(state.lrparam/float(M),state.lremb,XWNtrainl[:,XWNbatchct*M:(XWNbatchct+1)*M],XWNtrainr[:,XWNbatchct*M:(XWNbatchct+1)*M],XWNtraino[:,XWNbatchct*M:(XWNbatchct+1)*M],XWNtrainln[:,XWNbatchct*M:(XWNbatchct+1)*M],XWNtrainrn[:,XWNbatchct*M:(XWNbatchct+1)*M],XWNtrainon[:,XWNbatchct*M:(XWNbatchct+1)*M])
                XWNleft += [resl[1]/float(M)]
                XWNright += [resl[2]/float(M)]
                XWNrela += [resl[3]/float(M)]
                XWNleftb += [resl[5]/float(M)]
                XWNrightb += [resl[6]/float(M)]
                XWNrelab += [resl[7]/float(M)]
                embeddings.norma()
                XWNbatchct += 1

        order = numpy.random.permutation(WNtrainl.shape[1])
        WNtrainl = WNtrainl[:,order]
        WNtrainr = WNtrainr[:,order]
        WNtraino = WNtraino[:,order]
        WNtrainln = WNtrainln[:,numpy.random.permutation(WNtrainln.shape[1])]
        WNtrainrn = WNtrainrn[:,numpy.random.permutation(WNtrainln.shape[1])]
        WNtrainon = WNtrainon[:,numpy.random.permutation(WNtrainln.shape[1])]
        ct = ct + 1
        print >> sys.stderr, "FINISHED EPOCH %s --- current time: %s"%(ct,time.time()-ref)
        if ct/float(state.testall) == ct / state.testall:
            txt = ''
            txt += '------ Epoch %s ------ lr emb: %s ------ lr param: %s ------ time spent: %s\n'%(ct,state.lremb/float(1+state.ratdeclr*(ct-state.begindeclr)),state.lrparam/float(1+state.ratdeclr*(ct-state.begindeclr)),time.time()-ref)
            if state.updateWN: 
                txt += 'WN\n'
                txt += '%s %s %s %s %s %s %s %s\n'%(numpy.mean(WNleft+WNright+WNrela), numpy.std(WNleft+WNright+WNrela),numpy.mean(WNleft),numpy.std(WNleft),numpy.mean(WNright), numpy.std(WNright),numpy.mean(WNrela), numpy.std(WNrela))
                txt += '%s %s %s %s %s %s %s %s\n'%(numpy.mean(WNleftb+WNrightb+WNrelab), numpy.std(WNleftb+WNrightb+WNrelab),numpy.mean(WNleftb),numpy.std(WNleftb),numpy.mean(WNrightb), numpy.std(WNrightb),numpy.mean(WNrelab), numpy.std(WNrelab))
            if state.updateWNl:
                txt += 'WNl\n'
                txt += '%s %s %s %s %s %s %s %s\n'%(numpy.mean(WNlleft+WNlright+WNlrela), numpy.std(WNlleft+WNlright+WNlrela),numpy.mean(WNlleft),numpy.std(WNlleft),numpy.mean(WNlright), numpy.std(WNlright),numpy.mean(WNlrela), numpy.std(WNlrela))
                txt += '%s %s %s %s %s %s %s %s\n'%(numpy.mean(WNlleftb+WNlrightb+WNlrelab), numpy.std(WNlleftb+WNlrightb+WNlrelab),numpy.mean(WNlleftb),numpy.std(WNlleftb),numpy.mean(WNlrightb), numpy.std(WNlrightb),numpy.mean(WNlrelab), numpy.std(WNlrelab))
            if state.updateWNsl:
                txt += 'WNsl\n'
                txt += '%s %s %s %s %s %s %s %s\n'%(numpy.mean(WNslleft+WNslright+WNslrela), numpy.std(WNslleft+WNslright+WNslrela),numpy.mean(WNslleft),numpy.std(WNslleft),numpy.mean(WNslright), numpy.std(WNslright),numpy.mean(WNslrela), numpy.std(WNslrela))
                txt += '%s %s %s %s %s %s %s %s\n'%(numpy.mean(WNslleftb+WNslrightb+WNslrelab), numpy.std(WNslleftb+WNslrightb+WNslrelab),numpy.mean(WNslleftb),numpy.std(WNslleftb),numpy.mean(WNslrightb), numpy.std(WNslrightb),numpy.mean(WNslrelab), numpy.std(WNslrelab))
            if state.updateCN:
                txt += 'CN\n'
                txt += '%s %s %s %s %s %s %s %s\n'%(numpy.mean(CNleft+CNright+CNrela), numpy.std(CNleft+CNright+CNrela),numpy.mean(CNleft),numpy.std(CNleft),numpy.mean(CNright), numpy.std(CNright),numpy.mean(CNrela), numpy.std(CNrela))
                txt += '%s %s %s %s %s %s %s %s\n'%(numpy.mean(CNleftb+CNrightb+CNrelab), numpy.std(CNleftb+CNrightb+CNrelab),numpy.mean(CNleftb),numpy.std(CNleftb),numpy.mean(CNrightb), numpy.std(CNrightb),numpy.mean(CNrelab), numpy.std(CNrelab))
            if state.updateWK:
                txt += 'WK\n'
                txt += '%s %s %s %s %s %s %s %s\n'%(numpy.mean(WKleft+WKright+WKrela), numpy.std(WKleft+WKright+WKrela),numpy.mean(WKleft),numpy.std(WKleft),numpy.mean(WKright), numpy.std(WKright),numpy.mean(WKrela), numpy.std(WKrela))
                txt += '%s %s %s %s %s %s %s %s\n'%(numpy.mean(WKleftb+WKrightb+WKrelab), numpy.std(WKleftb+WKrightb+WKrelab),numpy.mean(WKleftb),numpy.std(WKleftb),numpy.mean(WKrightb), numpy.std(WKrightb),numpy.mean(WKrelab), numpy.std(WKrelab))
            if state.updateWKs:
                txt += 'WKs\n'
                txt += '%s %s %s %s %s %s %s %s\n'%(numpy.mean(WKsleft+WKsright+WKsrela), numpy.std(WKsleft+WKsright+WKsrela),numpy.mean(WKsleft),numpy.std(WKsleft),numpy.mean(WKsright), numpy.std(WKsright),numpy.mean(WKsrela), numpy.std(WKsrela))
                txt += '%s %s %s %s %s %s %s %s\n'%(numpy.mean(WKsleftb+WKsrightb+WKsrelab), numpy.std(WKsleftb+WKsrightb+WKsrelab),numpy.mean(WKsleftb),numpy.std(WKsleftb),numpy.mean(WKsrightb), numpy.std(WKsrightb),numpy.mean(WKsrelab), numpy.std(WKsrelab))
            if state.updateXWN:
                txt += 'XWN\n'
                txt += '%s %s %s %s %s %s %s %s\n'%(numpy.mean(XWNleft+XWNright+XWNrela), numpy.std(XWNleft+XWNright+XWNrela),numpy.mean(XWNleft),numpy.std(XWNleft),numpy.mean(XWNright), numpy.std(XWNright),numpy.mean(XWNrela), numpy.std(XWNrela))
                txt += '%s %s %s %s %s %s %s %s\n'%(numpy.mean(XWNleftb+XWNrightb+XWNrelab), numpy.std(XWNleftb+XWNrightb+XWNrelab),numpy.mean(XWNleftb),numpy.std(XWNleftb),numpy.mean(XWNrightb), numpy.std(XWNrightb),numpy.mean(XWNrelab), numpy.std(XWNrelab))

            dictparam['epochl'] += [ct]
            if ct > state.begindeclr:
                dictparam['lrparaml'] += [state.lrparam/float(1+state.ratdeclr * (ct-state.begindeclr))]
                dictparam['lrembl'] += [state.lremb/float(1+state.ratdeclr * (ct-state.begindeclr))]
            else:
                dictparam['lrparaml'] += [state.lrparam]
                dictparam['lrembl'] += [state.lremb]
            if state.updateWN:
                dictparam['WNallmean'] += [numpy.mean(WNleft+WNright+WNrela)]
                dictparam['WNallstd']  += [numpy.std(WNleft+WNright+WNrela)]
                dictparam['WNleftmean'] += [numpy.mean(WNleft)]
                dictparam['WNleftstd'] += [numpy.std(WNleft)]
                dictparam['WNrightmean'] += [numpy.mean(WNright)]
                dictparam['WNrightstd'] += [numpy.std(WNright)]
                dictparam['WNrelamean'] += [numpy.mean(WNrela)]
                dictparam['WNrelastd'] += [numpy.std(WNrela)]
                dictparam['WNallbmean'] += [numpy.mean(WNleftb+WNrightb+WNrelab)]
                dictparam['WNallbstd']  += [numpy.std(WNleftb+WNrightb+WNrelab)]
                dictparam['WNleftbmean'] += [numpy.mean(WNleftb)]
                dictparam['WNleftbstd'] += [numpy.std(WNleftb)]
                dictparam['WNrightbmean'] += [numpy.mean(WNrightb)]
                dictparam['WNrightbstd'] += [numpy.std(WNrightb)]
                dictparam['WNrelabmean'] += [numpy.mean(WNrelab)]
                dictparam['WNrelabstd'] += [numpy.std(WNrelab)]
                state.WNallmean = numpy.mean(WNleft+WNright+WNrela)
                state.WNallmeanb = numpy.mean(WNleftb+WNrightb+WNrelab)
            if state.updateWNl:
                dictparam['WNlallmean'] += [numpy.mean(WNlleft+WNlright+WNlrela)]
                dictparam['WNlallstd']  += [numpy.std(WNlleft+WNlright+WNlrela)]
                dictparam['WNlleftmean'] += [numpy.mean(WNlleft)]
                dictparam['WNlleftstd'] += [numpy.std(WNlleft)]
                dictparam['WNlrightmean'] += [numpy.mean(WNlright)]
                dictparam['WNlrightstd'] += [numpy.std(WNlright)]
                dictparam['WNlrelamean'] += [numpy.mean(WNlrela)]
                dictparam['WNlrelastd'] += [numpy.std(WNlrela)]
                dictparam['WNlallbmean'] += [numpy.mean(WNlleftb+WNlrightb+WNlrelab)]
                dictparam['WNlallbstd']  += [numpy.std(WNlleftb+WNlrightb+WNlrelab)]
                dictparam['WNlleftbmean'] += [numpy.mean(WNlleftb)]
                dictparam['WNlleftbstd'] += [numpy.std(WNlleftb)]
                dictparam['WNlrightbmean'] += [numpy.mean(WNlrightb)]
                dictparam['WNlrightbstd'] += [numpy.std(WNlrightb)]
                dictparam['WNlrelabmean'] += [numpy.mean(WNlrelab)]
                dictparam['WNlrelabstd'] += [numpy.std(WNlrelab)]
                state.WNlallmean = numpy.mean(WNlleft+WNlright+WNlrela)
                state.WNlallmeanb = numpy.mean(WNlleftb+WNlrightb+WNlrelab)
            if state.updateWNsl:
                dictparam['WNslallmean'] += [numpy.mean(WNslleft+WNslright+WNslrela)]
                dictparam['WNslallstd']  += [numpy.std(WNslleft+WNslright+WNslrela)]
                dictparam['WNslleftmean'] += [numpy.mean(WNslleft)]
                dictparam['WNslleftstd'] += [numpy.std(WNslleft)]
                dictparam['WNslrightmean'] += [numpy.mean(WNslright)]
                dictparam['WNslrightstd'] += [numpy.std(WNslright)]
                dictparam['WNslrelamean'] += [numpy.mean(WNslrela)]
                dictparam['WNslrelastd'] += [numpy.std(WNslrela)]
                dictparam['WNslallbmean'] += [numpy.mean(WNslleftb+WNslrightb+WNslrelab)]
                dictparam['WNslallbstd']  += [numpy.std(WNslleftb+WNslrightb+WNslrelab)]
                dictparam['WNslleftbmean'] += [numpy.mean(WNslleftb)]
                dictparam['WNslleftbstd'] += [numpy.std(WNslleftb)]
                dictparam['WNslrightbmean'] += [numpy.mean(WNslrightb)]
                dictparam['WNslrightbstd'] += [numpy.std(WNslrightb)]
                dictparam['WNslrelabmean'] += [numpy.mean(WNslrelab)]
                dictparam['WNslrelabstd'] += [numpy.std(WNslrelab)]
                state.WNslallmean = numpy.mean(WNslleft+WNslright+WNslrela)
                state.WNslallmeanb = numpy.mean(WNslleftb+WNslrightb+WNslrelab)
            if state.updateCN:
                dictparam['CNallmean'] += [numpy.mean(CNleft+CNright+CNrela)]
                dictparam['CNallstd']  += [numpy.std(CNleft+CNright+CNrela)]
                dictparam['CNleftmean'] += [numpy.mean(CNleft)]
                dictparam['CNleftstd'] += [numpy.std(CNleft)]
                dictparam['CNrightmean'] += [numpy.mean(CNright)]
                dictparam['CNrightstd'] += [numpy.std(CNright)]
                dictparam['CNrelamean'] += [numpy.mean(CNrela)]
                dictparam['CNrelastd'] += [numpy.std(CNrela)]
                dictparam['CNallbmean'] += [numpy.mean(CNleftb+CNrightb+CNrelab)]
                dictparam['CNallbstd']  += [numpy.std(CNleftb+CNrightb+CNrelab)]
                dictparam['CNleftbmean'] += [numpy.mean(CNleftb)]
                dictparam['CNleftbstd'] += [numpy.std(CNleftb)]
                dictparam['CNrightbmean'] += [numpy.mean(CNrightb)]
                dictparam['CNrightbstd'] += [numpy.std(CNrightb)]
                dictparam['CNrelabmean'] += [numpy.mean(CNrelab)]
                dictparam['CNrelabstd'] += [numpy.std(CNrelab)]
                state.CNallmean = numpy.mean(CNleft+CNright+CNrela)
                state.CNallmeanb = numpy.mean(CNleftb+CNrightb+CNrelab)
            if state.updateWK:
                dictparam['WKallmean'] += [numpy.mean(WKleft+WKright+WKrela)]
                dictparam['WKallstd']  += [numpy.std(WKleft+WKright+WKrela)]
                dictparam['WKleftmean'] += [numpy.mean(WKleft)]
                dictparam['WKleftstd'] += [numpy.std(WKleft)]
                dictparam['WKrightmean'] += [numpy.mean(WKright)]
                dictparam['WKrightstd'] += [numpy.std(WKright)]
                dictparam['WKrelamean'] += [numpy.mean(WKrela)]
                dictparam['WKrelastd'] += [numpy.std(WKrela)]
                dictparam['WKallbmean'] += [numpy.mean(WKleftb+WKrightb+WKrelab)]
                dictparam['WKallbstd']  += [numpy.std(WKleftb+WKrightb+WKrelab)]
                dictparam['WKleftbmean'] += [numpy.mean(WKleftb)]
                dictparam['WKleftbstd'] += [numpy.std(WKleftb)]
                dictparam['WKrightbmean'] += [numpy.mean(WKrightb)]
                dictparam['WKrightbstd'] += [numpy.std(WKrightb)]
                dictparam['WKrelabmean'] += [numpy.mean(WKrelab)]
                dictparam['WKrelabstd'] += [numpy.std(WKrelab)]
                state.WKallmean = numpy.mean(WKleft+WKright+WKrela)
                state.WKallmeanb = numpy.mean(WKleftb+WKrightb+WKrelab)
            if state.updateWKs:
                dictparam['WKsallmean'] += [numpy.mean(WKsleft+WKsright+WKsrela)]
                dictparam['WKsallstd']  += [numpy.std(WKsleft+WKsright+WKsrela)]
                dictparam['WKsleftmean'] += [numpy.mean(WKsleft)]
                dictparam['WKsleftstd'] += [numpy.std(WKsleft)]
                dictparam['WKsrightmean'] += [numpy.mean(WKsright)]
                dictparam['WKsrightstd'] += [numpy.std(WKsright)]
                dictparam['WKsrelamean'] += [numpy.mean(WKsrela)]
                dictparam['WKsrelastd'] += [numpy.std(WKsrela)]
                dictparam['WKsallbmean'] += [numpy.mean(WKsleftb+WKsrightb+WKsrelab)]
                dictparam['WKsallbstd']  += [numpy.std(WKsleftb+WKsrightb+WKsrelab)]
                dictparam['WKsleftbmean'] += [numpy.mean(WKsleftb)]
                dictparam['WKsleftbstd'] += [numpy.std(WKsleftb)]
                dictparam['WKsrightbmean'] += [numpy.mean(WKsrightb)]
                dictparam['WKsrightbstd'] += [numpy.std(WKsrightb)]
                dictparam['WKsrelabmean'] += [numpy.mean(WKsrelab)]
                dictparam['WKsrelabstd'] += [numpy.std(WKsrelab)]
                state.WKsallmean = numpy.mean(WKsleft+WKsright+WKsrela)
                state.WKsallmeanb = numpy.mean(WKsleftb+WKsrightb+WKsrelab)
            if state.updateXWN:
                dictparam['XWNallmean'] += [numpy.mean(XWNleft+XWNright+XWNrela)]
                dictparam['XWNallstd']  += [numpy.std(XWNleft+XWNright+XWNrela)]
                dictparam['XWNleftmean'] += [numpy.mean(XWNleft)]
                dictparam['XWNleftstd'] += [numpy.std(XWNleft)]
                dictparam['XWNrightmean'] += [numpy.mean(XWNright)]
                dictparam['XWNrightstd'] += [numpy.std(XWNright)]
                dictparam['XWNrelamean'] += [numpy.mean(XWNrela)]
                dictparam['XWNrelastd'] += [numpy.std(XWNrela)]
                dictparam['XWNallbmean'] += [numpy.mean(XWNleftb+XWNrightb+XWNrelab)]
                dictparam['XWNallbstd']  += [numpy.std(XWNleftb+XWNrightb+XWNrelab)]
                dictparam['XWNleftbmean'] += [numpy.mean(XWNleftb)]
                dictparam['XWNleftbstd'] += [numpy.std(XWNleftb)]
                dictparam['XWNrightbmean'] += [numpy.mean(XWNrightb)]
                dictparam['XWNrightbstd'] += [numpy.std(XWNrightb)]
                dictparam['XWNrelabmean'] += [numpy.mean(XWNrelab)]
                dictparam['XWNrelabstd'] += [numpy.std(XWNrelab)]
                state.XWNallmean = numpy.mean(XWNleft+XWNright+XWNrela)
                state.XWNallmeanb = numpy.mean(XWNleftb+XWNrightb+XWNrelab)

            WNleft = []
            WNright = []
            WNrela = []
            WNleftb = []
            WNrightb = []
            WNrelab = []

            WNlleft = []
            WNlright = []
            WNlrela = []
            WNlleftb = []
            WNlrightb = []
            WNlrelab = []

            WNslleft = []
            WNslright = []
            WNslrela = []
            WNslleftb = []
            WNslrightb = []
            WNslrelab = []

            CNleft = []
            CNright = []
            CNrela = []
            CNleftb = []
            CNrightb = []
            CNrelab = []

            WKleft = []
            WKright = []
            WKrela = []
            WKleftb = []
            WKrightb = []
            WKrelab = []

            WKsleft = []
            WKsright = []
            WKsrela = []
            WKsleftb = []
            WKsrightb = []
            WKsrelab = []

            XWNleft = []
            XWNright = []
            XWNrela = []
            XWNleftb = []
            XWNrightb = []
            XWNrelab = []
            
            if state.updateWN:
                resultt = calctestval(sl,sr,idxtl[:state.nbtest],idxtr[:state.nbtest],idxto[:state.nbtest])
                resultv = calctestval(sl,sr,idxvl[:state.nbtest],idxvr[:state.nbtest],idxvo[:state.nbtest])
                state.WNval = resultv[0]
                if state.bestWNval == -1 or state.WNval<state.bestWNval:
                    state.bestWNval = state.WNval
                state.WNtes = resultt[0]
                if state.bestWNtes == -1 or state.WNtes<state.bestWNtes:
                    state.bestWNtes = state.WNtes
                txt += 'WN:\n'
                txt += 'test' + str(resultt)+'\n'
                txt += 'val' + str(resultv)+'\n'
            if state.updateWK:
                resl = vt(WKtrainvall,WKtrainvalr,WKtrainvalo,WKtrainvalln,WKtrainvalrn,WKtrainvalon)
                left = [resl[1]/float(WKtrainvall.shape[1])]
                right = [resl[2]/float(WKtrainvall.shape[1])]
                rela = [resl[3]/float(WKtrainvall.shape[1])]
                leftb = [resl[5]/float(WKtrainvall.shape[1])]
                rightb = [resl[6]/float(WKtrainvall.shape[1])]
                relab = [resl[7]/float(WKtrainvall.shape[1])]
                leftm = [resl[8]/float(WKtrainvall.shape[1])]
                rightm = [resl[9]/float(WKtrainvall.shape[1])]
                relam = [resl[10]/float(WKtrainvall.shape[1])]
                state.WKval = numpy.mean(left+right+rela)
                if state.bestWKval == -1 or state.WKval<state.bestWKval:
                    state.bestWKval = state.WKval
                state.WKvalb = numpy.mean(leftb+rightb+relab)
                if state.bestWKvalb == -1 or state.WKvalb<state.bestWKvalb:
                    state.bestWKvalb = state.WKvalb
                state.WKvalm = numpy.mean(leftm+rightm+relam)
                if state.bestWKvalm == -1 or state.WKvalm<state.bestWKvalm:
                    state.bestWKvalm = state.WKvalm
                dictparam['WKval'] += [numpy.mean(left+right+rela)]
                dictparam['WKvalb'] += [numpy.mean(leftb+rightb+relab)]
                dictparam['WKvalm'] += [numpy.mean(leftm+rightm+relam)]
                txt += 'WKtrain:\n'
                txt += '%s %s %s %s\n'%(numpy.mean(left+right+rela),numpy.mean(left),numpy.mean(right),numpy.mean(rela)) 
                txt += '%s %s %s %s\n'%(numpy.mean(leftb+rightb+relab),numpy.mean(leftb),numpy.mean(rightb),numpy.mean(relab))
                txt += '%s %s %s %s\n'%(numpy.mean(leftm+rightm+relam),numpy.mean(leftm),numpy.mean(rightm),numpy.mean(relam))
            if state.updateWKs:
                resl = vt(WKstrainvall,WKstrainvalr,WKstrainvalo,WKstrainvalln,WKstrainvalrn,WKstrainvalon)
                left = [resl[1]/float(WKstrainvall.shape[1])]
                right = [resl[2]/float(WKstrainvall.shape[1])]
                rela = [resl[3]/float(WKstrainvall.shape[1])]
                leftb = [resl[5]/float(WKstrainvall.shape[1])]
                rightb = [resl[6]/float(WKstrainvall.shape[1])]
                relab = [resl[7]/float(WKstrainvall.shape[1])]
                leftm = [resl[8]/float(WKstrainvall.shape[1])]
                rightm = [resl[9]/float(WKstrainvall.shape[1])]
                relam = [resl[10]/float(WKstrainvall.shape[1])]
                state.WKsval = numpy.mean(left+right+rela)
                if state.bestWKsval == -1 or state.WKsval<state.bestWKsval:
                    state.bestWKsval = state.WKsval
                state.WKsvalb = numpy.mean(leftb+rightb+relab)
                if state.bestWKsvalb == -1 or state.WKsvalb<state.bestWKsvalb:
                    state.bestWKsvalb = state.WKsvalb
                state.WKsvalm = numpy.mean(leftm+rightm+relam)
                if state.bestWKsvalm == -1 or state.WKsvalm<state.bestWKsvalm:
                    state.bestWKsvalm = state.WKsvalm
                dictparam['WKsval'] += [numpy.mean(left+right+rela)]
                dictparam['WKsvalb'] += [numpy.mean(leftb+rightb+relab)]
                dictparam['WKsvalm'] += [numpy.mean(leftm+rightm+relam)]
                txt += 'WKstrain:\n'
                txt += '%s %s %s %s\n'%(numpy.mean(left+right+rela),numpy.mean(left),numpy.mean(right),numpy.mean(rela))
                txt += '%s %s %s %s\n'%(numpy.mean(leftb+rightb+relab),numpy.mean(leftb),numpy.mean(rightb),numpy.mean(relab))
                txt += '%s %s %s %s\n'%(numpy.mean(leftm+rightm+relam),numpy.mean(leftm),numpy.mean(rightm),numpy.mean(relam))
            if state.updateXWN:
                resl = vt(XWNtrainvall,XWNtrainvalr,XWNtrainvalo,XWNtrainvalln,XWNtrainvalrn,XWNtrainvalon)
                left = [resl[1]/float(XWNtrainvall.shape[1])]
                right = [resl[2]/float(XWNtrainvall.shape[1])]
                rela = [resl[3]/float(XWNtrainvall.shape[1])]
                leftb = [resl[5]/float(XWNtrainvall.shape[1])]
                rightb = [resl[6]/float(XWNtrainvall.shape[1])]
                relab = [resl[7]/float(XWNtrainvall.shape[1])]
                leftm = [resl[8]/float(XWNtrainvall.shape[1])]
                rightm = [resl[9]/float(XWNtrainvall.shape[1])]
                relam = [resl[10]/float(XWNtrainvall.shape[1])]
                state.XWNval = numpy.mean(left+right+rela)
                if state.bestXWNval == -1 or state.XWNval<state.bestXWNval:
                    state.bestXWNval = state.XWNval
                state.XWNvalb = numpy.mean(leftb+rightb+relab)
                if state.bestXWNvalb == -1 or state.XWNvalb<state.bestXWNvalb:
                    state.bestXWNvalb = state.XWNvalb
                state.XWNvalm = numpy.mean(leftm+rightm+relam)
                if state.bestXWNvalm == -1 or state.XWNvalm<state.bestXWNvalm:
                    state.bestXWNvalm = state.XWNvalm
                dictparam['XWNval'] += [numpy.mean(left+right+rela)]
                dictparam['XWNvalb'] += [numpy.mean(leftb+rightb+relab)]
                dictparam['XWNvalm'] += [numpy.mean(leftm+rightm+relam)]
                txt += 'XWNtrain:\n'
                txt += '%s %s %s %s\n'%(numpy.mean(left+right+rela),numpy.mean(left),numpy.mean(right),numpy.mean(rela))
                txt += '%s %s %s %s\n'%(numpy.mean(leftb+rightb+relab),numpy.mean(leftb),numpy.mean(rightb),numpy.mean(relab))
                txt += '%s %s %s %s\n'%(numpy.mean(leftm+rightm+relam),numpy.mean(leftm),numpy.mean(rightm),numpy.mean(relam))
            txtall = txt
            for cc in state.listconcept:
                txtall+='\n'
                txtall += getnclosest(state.nbrank, idx2lemme, lemme2idx, idx2synset, synset2idx, synset2def, synset2concept, concept2synset, Esim, cc, [], typ = 0, emb = True)
                for rr in state.listrel:
                    txtall+='\n'
                    txtall += getnclosest(state.nbrank,idx2lemme, lemme2idx, idx2synset, synset2idx, synset2def, synset2concept, concept2synset , sll, cc, rr, typ = 1, emb = False)
                    txtall+='\n'
                    txtall += getnclosest(state.nbrank,idx2lemme, lemme2idx, idx2synset, synset2idx, synset2def, synset2concept, concept2synset , srl, cc, rr, typ = 2, emb = False)
                for rr in state.listconcept:
                    txtall +='\n'
                    txtall += getnclosest(state.nbrank,idx2lemme, lemme2idx, idx2synset, synset2idx, synset2def, synset2concept, concept2synset , srl, cc, rr, typ = 3, emb = False)
            f = open(state.savepath+'/model.pkl','w')
            cPickle.dump(embeddings,f,-1)
            cPickle.dump(leftop,f,-1)
            cPickle.dump(rightop,f,-1)
            if state.simfnstr == 'MLP':
                cPickle.dump(MLPout,f,-1)
            f.close()
            f = open(state.savepath+'/currentrel.txt','w')
            f.write(txtall)
            f.close()
            f = open(state.savepath+'/log.txt','a')
            f.write(txt)
            f.close()
            f = open(state.savepath+'/modeldict.pkl','w')
            cPickle.dump(dictparam,f,-1)
            f.close()
            print >> sys.stderr, txt 
            ref = time.time()
            state.nbupdates = ct * state.nbatches
            state.nbexamples = ct * state.nbatches * M
            state.nbepochs = ct
            channel.save()

