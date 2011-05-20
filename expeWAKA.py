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
    datpath = '/mnt/scratch/bengio/glorotxa/data/exp/WakaBST3/'

    state.listconcept = [['__brain_NN_1'],  ['__france_NN_1'], ['__auto_NN_1'],['__cat_NN_1'],['__monkey_NN_1'],['__u.s._NN_1','__army_NN_1']]
    state.listrel = [['_has_part'],['_part_of'],['__eat_VB_1'],['__drive_VB_1'],['__defend_VB_1'],['__attack_VB_1']]  
 
    dictparam = {}
    dictparam.update({ 'operator':state.operator})
    dictparam.update({ 'updateWN' :  state.updateWN})
    dictparam.update({ 'updateWKl' : state.updateWKl})
    dictparam.update({ 'updateWKs' : state.updateWKs})
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
    dictparam.update({ 'margel' : state.margel})
    dictparam.update({ 'marges' : state.marges})
    dictparam.update({ 'relb' : state.relb})
    dictparam.update({ 'simpleWN' : state.simpleWN})
    dictparam.update({ 'random' : state.random})
    
    
    print >> sys.stderr, 'operator : ', state.operator
    print >> sys.stderr, 'updateWN : ', state.updateWN
    print >> sys.stderr, 'updateWKl : ', state.updateWKl
    print >> sys.stderr, 'updateWKs : ', state.updateWKs
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
    print >> sys.stderr, 'margel: ', state.margel
    print >> sys.stderr, 'marges: ', state.marges
    print >> sys.stderr, 'relb: ', state.relb
    print >> sys.stderr, 'simpleWN: ', state.simpleWN
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

    ####### WORDNET
    # train set
    if state.simpleWN:
        WNtrainl = (cPickle.load(open(datpath+'WordNet3.0-easy-train-lhs.pkl','r')))
        WNtrainr = (cPickle.load(open(datpath+'WordNet3.0-easy-train-rhs.pkl','r')))
        WNtraino = (cPickle.load(open(datpath+'WordNet3.0-easy-train-rel.pkl','r')))
    else:
        WNtrainl = (cPickle.load(open(datpath+'WordNet3.0-train-lhs.pkl','r')))
        WNtrainr = (cPickle.load(open(datpath+'WordNet3.0-train-rhs.pkl','r')))
        WNtraino = (cPickle.load(open(datpath+'WordNet3.0-train-rel.pkl','r')))

    if not state.random:
        WNtrainln = (cPickle.load(open(datpath+'WordNet3.0-train-lhs.pkl','r')))
        WNtrainrn = (cPickle.load(open(datpath+'WordNet3.0-train-rhs.pkl','r')))
        WNtrainon = (cPickle.load(open(datpath+'WordNet3.0-train-rel.pkl','r')))
    else:
        WNtrainln = createrandommat(WNtrainl.shape)
        WNtrainrn = createrandommat(WNtrainl.shape)
        WNtrainon = createrandommat(WNtrainl.shape)

    numpy.random.seed(111)
    order = numpy.random.permutation(WNtrainl.shape[1])
    WNtrainl = WNtrainl[:,order]
    WNtrainr = WNtrainr[:,order]
    WNtraino = WNtraino[:,order]
    WNtrainln = WNtrainln[:,numpy.random.permutation(WNtrainln.shape[1])]
    WNtrainrn = WNtrainrn[:,numpy.random.permutation(WNtrainln.shape[1])]
    WNtrainon = WNtrainon[:,numpy.random.permutation(WNtrainln.shape[1])]
    
    # valid set
    WNvall = (cPickle.load(open(datpath+'WordNet3.0-easy-val-lhs.pkl','r')))
    WNvalr = (cPickle.load(open(datpath+'WordNet3.0-easy-val-rhs.pkl','r')))
    WNvalo = (cPickle.load(open(datpath+'WordNet3.0-easy-val-rel.pkl','r')))
    numpy.random.seed(222)
    order = numpy.random.permutation(WNvall.shape[1])
    WNvall = WNvall[:,order]
    WNvalr = WNvalr[:,order]
    WNvalo = WNvalo[:,order]

    # test set
    WNtestl = (cPickle.load(open(datpath+'WordNet3.0-easy-test-lhs.pkl','r')))
    WNtestr = (cPickle.load(open(datpath+'WordNet3.0-easy-test-rhs.pkl','r')))
    WNtesto = (cPickle.load(open(datpath+'WordNet3.0-easy-test-rel.pkl','r')))
    numpy.random.seed(333)
    order = numpy.random.permutation(WNtestl.shape[1])
    WNtestl = WNtestl[:,order]
    WNtestr = WNtestr[:,order]
    WNtesto = WNtesto[:,order]

    ###### Wikilemmes
    WKlemmel = (cPickle.load(open(datpath+'Wikilemmes-lhs.pkl','r')))
    WKlemmer = (cPickle.load(open(datpath+'Wikilemmes-rhs.pkl','r')))
    WKlemmeo = (cPickle.load(open(datpath+'Wikilemmes-rel.pkl','r')))
    
    if not state.random:
        WKlemmeln = (cPickle.load(open(datpath+'Wikilemmes-lhs.pkl','r')))
        WKlemmern = (cPickle.load(open(datpath+'Wikilemmes-rhs.pkl','r')))
        WKlemmeon = (cPickle.load(open(datpath+'Wikilemmes-rel.pkl','r')))
    else:
        WKlemmeln = createrandommat(WKlemmel.shape)
        WKlemmern = createrandommat(WKlemmel.shape)
        WKlemmeon = createrandommat(WKlemmel.shape)
    
    numpy.random.seed(444)
    order = numpy.random.permutation(WKlemmel.shape[1])
    WKlemmel = WKlemmel[:,order]
    WKlemmer = WKlemmer[:,order]
    WKlemmeo = WKlemmeo[:,order]
    WKlemmeln = WKlemmeln[:,numpy.random.permutation(WKlemmeln.shape[1])]
    WKlemmern = WKlemmern[:,numpy.random.permutation(WKlemmeln.shape[1])]
    WKlemmeon = WKlemmeon[:,numpy.random.permutation(WKlemmeln.shape[1])]
    
    WKlemmevall = WKlemmel[:,-10000:]
    WKlemmevalr = WKlemmer[:,-10000:]
    WKlemmevalo = WKlemmeo[:,-10000:]
    WKlemmevalln = WKlemmeln[:,-10000:]
    WKlemmevalrn = WKlemmern[:,-10000:]
    WKlemmevalon = WKlemmeon[:,-10000:]
    
    WKlemmel = WKlemmel[:,:-10000] 
    WKlemmer = WKlemmer[:,:-10000]
    WKlemmeo = WKlemmeo[:,:-10000]
    WKlemmeln = WKlemmeln[:,:-10000]
    WKlemmern = WKlemmern[:,:-10000]
    WKlemmeon = WKlemmeon[:,:-10000]


    ###### Wikisuper
    WKsuperl = (cPickle.load(open(datpath+'Wikisuper-lhs.pkl','r')))
    WKsuperr = (cPickle.load(open(datpath+'Wikisuper-rhs.pkl','r')))
    WKsupero = (cPickle.load(open(datpath+'Wikisuper-rel.pkl','r')))

    WKsuperln = (cPickle.load(open(datpath+'Wikisuper-lhsn.pkl','r')))
    WKsuperrn = (cPickle.load(open(datpath+'Wikisuper-rhsn.pkl','r')))
    WKsuperon = (cPickle.load(open(datpath+'Wikisuper-reln.pkl','r')))

    WKsupervall = WKsuperl[:,-10000:]
    WKsupervalr = WKsuperr[:,-10000:]
    WKsupervalo = WKsupero[:,-10000:]
    WKsupervalln = WKsuperln[:,-10000:]
    WKsupervalrn = WKsuperrn[:,-10000:]
    WKsupervalon = WKsuperon[:,-10000:]
    WKsuperl = WKsuperl[:,:-10000]
    WKsuperr = WKsuperr[:,:-10000]
    WKsupero = WKsupero[:,:-10000]
    WKsuperln = WKsuperln[:,:-10000]
    WKsuperrn = WKsuperrn[:,:-10000]
    WKsuperon = WKsuperon[:,:-10000]
    # indcator for big test:
    state.bigtest = 1

    numpy.random.seed(555) 
    neworder = numpy.random.permutation(WKsuperln.shape[1])
    WKsuperln = WKsuperln[:,neworder]
    WKsuperrn = WKsuperrn[:,neworder]
    WKsuperon = WKsuperon[:,neworder]
    WKsuperl = WKsuperl[:,neworder]
    WKsuperr = WKsuperr[:,neworder]
    WKsupero = WKsupero[:,neworder]

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
        f = open(loadmodel)
        embeddings = cPickle.load(f)
        leftop = cPickle.load(f)
        rightop = cPickle.load(f)
        if state.simfnstr == 'MLP':
           MLPout = cPickle.load(f)
        f.close()
        dictparam = cPickle.load(open(loadmodel[:-4]+'dict.pkl'))
    
    if state.simfnstr == 'MLP':
        simfn = MLPout
    else:
        simfn = eval(state.simfnstr+'sim')


    # train function
    ft = TrainFunction(simfn,embeddings,leftop,rightop, marge = state.margewn, relb = state.relb)
    ftlemme = TrainFunction(simfn,embeddings,leftop,rightop,marge=state.margel, relb = state.relb)
    ftsuper = TrainFunction(simfn,embeddings,leftop,rightop,marge=state.marges, relb = state.relb)
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
        dictparam['WKlallmean'] = []
        dictparam['WKlallstd']  = []
        dictparam['WKleftmean'] = []
        dictparam['WKlleftstd'] = []
        dictparam['WKlrightmean'] = []
        dictparam['WKlrightstd'] = []
        dictparam['WKlrelamean'] = []
        dictparam['WKlrelastd'] = []
        dictparam['WKlallbmean'] = []
        dictparam['WKlallbstd']  = []
        dictparam['WKlleftbmean'] = []
        dictparam['WKlleftbstd'] = []
        dictparam['WKlrightbmean'] = []
        dictparam['WKlrightbstd'] = []
        dictparam['WKlrelabmean'] = []
        dictparam['WKlrelabstd'] = []
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
        dictparam['WNval'] = []
        dictparam['WNtes'] = []
        dictparam['WKlval'] = []
        dictparam['WKlvalb'] = []
        dictparam['WKlvalm'] = []
        dictparam['WKsval'] = []
        dictparam['WKsvalb'] = []
        dictparam['WKsvalm'] = []


    WNleft = []
    WNright = []
    WNrela = []
    WNleftb = []
    WNrightb = []
    WNrelab = []

    WKlleft = []
    WKlright = []
    WKlrela = []
    WKlleftb = []
    WKlrightb = []
    WKlrelab = []

    WKsleft = []
    WKsright = []
    WKsrela = []
    WKsleftb = []
    WKsrightb = []
    WKsrelab = []
    
    state.bestWNval = -1
    state.bestWNtes = -1
    state.bestWKlval = -1
    state.bestWKlvalm = -1
    state.bestWKlvalb = -1
    state.bestWKsval = -1
    state.bestWKsvalm = -1
    state.bestWKsvalb = -1

    M = WNtrainl.shape[1]/state.nbatches
    Wikilemmebatch = WKlemmeln.shape[1] / M
    Wikilbatchct=0
    Wikisuperbatch = WKsuperln.shape[1] / M
    Wikisbatchct=0

    ref = time.time()
    print >> sys.stderr, "BEGIN TRAINING"
    for ccc in range(state.totbatch): 
        for i in range(state.nbatches):
            if state.updateWN:
                if ct > state.begindeclr:
                    resl = ft(state.lrparam/float((1+state.ratdeclr * (ct-state.begindeclr))*float(M)),state.lremb/float(1+state.ratdeclr * (ct-state.begindeclr)),WNtrainl[:,i*M:(i+1)*M],WNtrainr[:,i*M:(i+1)*M],WNtraino[:,i*M:(i+1)*M],WNtrainln[:,i*M:(i+1)*M],WNtrainrn[:,i*M:(i+1)*M],WNtrainon[:,i*M:(i+1)*M])
                else:
                    resl = ft(state.lrparam/float(M),state.lremb,WNtrainl[:,i*M:(i+1)*M],WNtrainr[:,i*M:(i+1)*M],WNtraino[:,i*M:(i+1)*M],WNtrainln[:,i*M:(i+1)*M],WNtrainrn[:,i*M:(i+1)*M],WNtrainon[:,i*M:(i+1)*M])
                WNleft += [resl[1]/float(M)]
                WNright += [resl[2]/float(M)]
                WNrela += [resl[3]/float(M)]
                WNleftb += [resl[5]/float(M)]
                WNrightb += [resl[6]/float(M)]
                WNrelab += [resl[7]/float(M)]
                embeddings.norma()
            
            if state.updateWKl:
                if Wikilbatchct == Wikilemmebatch:
                    WKlemmeln = WKlemmeln[:,numpy.random.permutation(WKlemmeln.shape[1])]
                    WKlemmern = WKlemmern[:,numpy.random.permutation(WKlemmeln.shape[1])]
                    WKlemmeon = WKlemmeon[:,numpy.random.permutation(WKlemmeln.shape[1])]
                    neworder = numpy.random.permutation(WKlemmeln.shape[1])
                    WKlemmel = WKlemmel[:,neworder]
                    WKlemmer = WKlemmer[:,neworder]
                    WKlemmeo = WKlemmeo[:,neworder]
                    Wikilbatchct = 0
                if ct > state.begindeclr:
                    resl = ftlemme(state.lrparam/float((1+state.ratdeclr * (ct-state.begindeclr))*float(M)),state.lremb/float(1+state.ratdeclr * (ct-state.begindeclr)),WKlemmel[:,Wikilbatchct*M:(Wikilbatchct+1)*M],WKlemmer[:,Wikilbatchct*M:(Wikilbatchct+1)*M],WKlemmeo[:,Wikilbatchct*M:(Wikilbatchct+1)*M],WKlemmeln[:,Wikilbatchct*M:(Wikilbatchct+1)*M],WKlemmern[:,Wikilbatchct*M:(Wikilbatchct+1)*M],WKlemmeon[:,Wikilbatchct*M:(Wikilbatchct+1)*M])
                else:
                    resl = ftlemme(state.lrparam/float(M),state.lremb,WKlemmel[:,Wikilbatchct*M:(Wikilbatchct+1)*M],WKlemmer[:,Wikilbatchct*M:(Wikilbatchct+1)*M],WKlemmeo[:,Wikilbatchct*M:(Wikilbatchct+1)*M],WKlemmeln[:,Wikilbatchct*M:(Wikilbatchct+1)*M],WKlemmern[:,Wikilbatchct*M:(Wikilbatchct+1)*M],WKlemmeon[:,Wikilbatchct*M:(Wikilbatchct+1)*M])
                WKlleft += [resl[1]/float(M)]
                WKlright += [resl[2]/float(M)]
                WKlrela += [resl[3]/float(M)]
                WKlleftb += [resl[5]/float(M)]
                WKlrightb += [resl[6]/float(M)]
                WKlrelab += [resl[7]/float(M)]
                embeddings.norma()
                Wikilbatchct += 1
            
            if state.updateWKs:
                if Wikisbatchct == Wikisuperbatch:
                    neworder = numpy.random.permutation(WKsuperln.shape[1])
                    WKsuperln = WKsuperln[:,neworder]
                    WKsuperrn = WKsuperrn[:,neworder]
                    WKsuperon = WKsuperon[:,neworder]
                    WKsuperl = WKsuperl[:,neworder]
                    WKsuperr = WKsuperr[:,neworder]
                    WKsupero = WKsupero[:,neworder]
                    Wikisbatchct = 0
                
                if ct > state.begindeclr:
                     resl = ftsuper(state.lrparam/float((1+state.ratdeclr * (ct-state.begindeclr))*float(M)),state.lremb/float(1+state.ratdeclr * (ct-state.begindeclr)),WKsuperl[:,Wikisbatchct*M:(Wikisbatchct+1)*M],WKsuperr[:,Wikisbatchct*M:(Wikisbatchct+1)*M],WKsupero[:,Wikisbatchct*M:(Wikisbatchct+1)*M],WKsuperln[:,Wikisbatchct*M:(Wikisbatchct+1)*M],WKsuperrn[:,Wikisbatchct*M:(Wikisbatchct+1)*M],WKsuperon[:,Wikisbatchct*M:(Wikisbatchct+1)*M])
                else:
                     resl = ftsuper(state.lrparam/float(M),state.lremb,WKsuperl[:,Wikisbatchct*M:(Wikisbatchct+1)*M],WKsuperr[:,Wikisbatchct*M:(Wikisbatchct+1)*M],WKsupero[:,Wikisbatchct*M:(Wikisbatchct+1)*M],WKsuperln[:,Wikisbatchct*M:(Wikisbatchct+1)*M],WKsuperrn[:,Wikisbatchct*M:(Wikisbatchct+1)*M],WKsuperon[:,Wikisbatchct*M:(Wikisbatchct+1)*M])
                WKsleft += [resl[1]/float(M)]
                WKsright += [resl[2]/float(M)]
                WKsrela += [resl[3]/float(M)]
                WKsleftb += [resl[5]/float(M)]
                WKsrightb += [resl[6]/float(M)]
                WKsrelab += [resl[7]/float(M)]
                embeddings.norma()
                Wikisbatchct += 1

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
            if state.updateWKl:
                txt += 'Wklemme\n'
                txt += '%s %s %s %s %s %s %s %s\n'%(numpy.mean(WKlleft+WKlright+WKlrela), numpy.std(WKlleft+WKlright+WKlrela),numpy.mean(WKlleft),numpy.std(WKlleft),numpy.mean(WKlright), numpy.std(WKlright),numpy.mean(WKlrela), numpy.std(WKlrela))
                txt += '%s %s %s %s %s %s %s %s\n'%( numpy.mean(WKlleftb+WKlrightb+WKlrelab), numpy.std(WKlleftb+WKlrightb+WKlrelab),numpy.mean(WKlleftb),numpy.std(WKlleftb),numpy.mean(WKlrightb), numpy.std(WKlrightb),numpy.mean(WKlrelab), numpy.std(WKlrelab))
            if state.updateWKs:
                txt += 'Wksuper\n'
                txt += '%s %s %s %s %s %s %s %s\n'%(numpy.mean(WKsleft+WKsright+WKsrela), numpy.std(WKsleft+WKsright+WKsrela),numpy.mean(WKsleft),numpy.std(WKsleft),numpy.mean(WKsright), numpy.std(WKsright),numpy.mean(WKsrela), numpy.std(WKsrela))
                txt += '%s %s %s %s %s %s %s %s\n'%( numpy.mean(WKsleftb+WKsrightb+WKsrelab), numpy.std(WKsleftb+WKsrightb+WKsrelab),numpy.mean(WKsleftb),numpy.std(WKsleftb),numpy.mean(WKsrightb), numpy.std(WKsrightb),numpy.mean(WKsrelab), numpy.std(WKsrelab))
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
            if state.updateWKl:
                dictparam['WKlallmean'] += [numpy.mean(WKlleft+WKlright+WKlrela)]
                dictparam['WKlallstd']  += [numpy.std(WKlleft+WKlright+WKlrela)]
                dictparam['WKleftmean'] += [numpy.mean(WKlleft)]
                dictparam['WKlleftstd'] += [numpy.std(WKlleft)]
                dictparam['WKlrightmean'] += [numpy.mean(WKlright)]
                dictparam['WKlrightstd'] += [numpy.std(WKlright)]
                dictparam['WKlrelamean'] += [numpy.mean(WKlrela)]
                dictparam['WKlrelastd'] += [numpy.std(WKlrela)]
                dictparam['WKlallbmean'] += [numpy.mean(WKlleftb+WKlrightb+WKlrelab)]
                dictparam['WKlallbstd']  += [numpy.std(WKlleftb+WKlrightb+WKlrelab)]
                dictparam['WKlleftbmean'] += [numpy.mean(WKlleftb)]
                dictparam['WKlleftbstd'] += [numpy.std(WKlleftb)]
                dictparam['WKlrightbmean'] += [numpy.mean(WKlrightb)]
                dictparam['WKlrightbstd'] += [numpy.std(WKlrightb)]
                dictparam['WKlrelabmean'] += [numpy.mean(WKlrelab)]
                dictparam['WKlrelabstd'] += [numpy.std(WKlrelab)]
                state.WKlallmean = numpy.mean(WKlleft+WKlright+WKlrela)
                state.WKlallmeanb = numpy.mean(WKlleftb+WKlrightb+WKlrelab)
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
            WNleft = []
            WNright = []
            WNrela = []
            WNleftb = []
            WNrightb = []
            WNrelab = []

            WKlleft = []
            WKlright = []
            WKlrela = []
            WKlleftb = []
            WKlrightb = []
            WKlrelab = []

            WKsleft = []
            WKsright = []
            WKsrela = []
            WKsleftb = []
            WKsrightb = []
            WKsrelab = []
            
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
            if state.updateWKl:
                resl = vt(WKlemmevall,WKlemmevalr,WKlemmevalo,WKlemmevalln,WKlemmevalrn,WKlemmevalon)
                left = [resl[1]/float(WKlemmevall.shape[1])]
                right = [resl[2]/float(WKlemmevall.shape[1])]
                rela = [resl[3]/float(WKlemmevall.shape[1])]
                leftb = [resl[5]/float(WKlemmevall.shape[1])]
                rightb = [resl[6]/float(WKlemmevall.shape[1])]
                relab = [resl[7]/float(WKlemmevall.shape[1])]
                leftm = [resl[8]/float(WKlemmevall.shape[1])]
                rightm = [resl[9]/float(WKlemmevall.shape[1])]
                relam = [resl[10]/float(WKlemmevall.shape[1])]
                state.WKlval = numpy.mean(left+right+rela)
                if state.bestWKlval == -1 or state.WKlval<state.bestWKlval:
                    state.bestWKlval = state.WKlval
                state.WKlvalb = numpy.mean(leftb+rightb+relab)
                if state.bestWKlvalb == -1 or state.WKlvalb<state.bestWKlvalb:
                    state.bestWKlvalb = state.WKlvalb
                state.WKlvalm = numpy.mean(leftm+rightm+relam)
                if state.bestWKlvalm == -1 or state.WKlvalm<state.bestWKlvalm:
                    state.bestWKlvalm = state.WKlvalm
                dictparam['WKlval'] += [numpy.mean(left+right+rela)]
                dictparam['WKlvalb'] += [numpy.mean(leftb+rightb+relab)]
                dictparam['WKlvalm'] += [numpy.mean(leftm+rightm+relam)]
                txt += 'WKlemme:\n'
                txt += '%s %s %s %s\n'%(numpy.mean(left+right+rela),numpy.mean(left),numpy.mean(right),numpy.mean(rela)) 
                txt += '%s %s %s %s\n'%(numpy.mean(leftb+rightb+relab),numpy.mean(leftb),numpy.mean(rightb),numpy.mean(relab))
                txt += '%s %s %s %s\n'%(numpy.mean(leftm+rightm+relam),numpy.mean(leftm),numpy.mean(rightm),numpy.mean(relam))
            if state.updateWKs:
                resl = vt(WKsupervall,WKsupervalr,WKsupervalo,WKsupervalln,WKsupervalrn,WKsupervalon)
                left = [resl[1]/float(WKsupervall.shape[1])]
                right = [resl[2]/float(WKsupervall.shape[1])]
                rela = [resl[3]/float(WKsupervall.shape[1])]
                leftb = [resl[5]/float(WKsupervall.shape[1])]
                rightb = [resl[6]/float(WKsupervall.shape[1])]
                relab = [resl[7]/float(WKsupervall.shape[1])]
                leftm = [resl[8]/float(WKsupervall.shape[1])]
                rightm = [resl[9]/float(WKsupervall.shape[1])]
                relam = [resl[10]/float(WKsupervall.shape[1])]
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
                txt += 'WKsuper:\n'
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

