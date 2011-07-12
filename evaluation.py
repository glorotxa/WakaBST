import scipy.sparse
import cPickle
import os
import sys
from model import *


# create or use a folder containing the results with the name:
name = sys.argv[1]

# take the job id of the experiment folder
id = int(sys.argv[2])

# do the evaluation given by the number (0 for all)
evaln = int(sys.argv[3])


def normafunc(x):
    return (x-numpy.min(x))/sum((x-numpy.min(x)))

def softmaxfunc(x):
    return numpy.exp(x-numpy.max(x))/sum(numpy.exp(x-numpy.max(x)))

print name,id,evaln

try:
    os.mkdir(name)
except:
    pass

datpath ='/data/lisa/exp/glorotxa/WakaBST4/'

synset2idx = cPickle.load(open(datpath+'synset2idx.pkl','r'))
lemme2idx = cPickle.load(open(datpath+'lemme2idx.pkl','r'))
loadmodel = '/data/lisa/exp/glorotxa/WakaBST4/evaluationsave/%s/model.pkl'%id


f = open(loadmodel)
embeddings = cPickle.load(f)
leftop = cPickle.load(f)
rightop = cPickle.load(f)
simfn = eval('dotsim')
try:
    MLPout = cPickle.load(f)
    simfn = MLPout
except:
    simfn = eval('dotsim')

#----------------------------------------------------------------------------------------------------
if evaln == 1 or evaln == 0:
    srl = SimilarityFunctionrightl(simfn,embeddings,leftop,rightop,numpy.max(synset2idx.values())+1,True)
    sll = SimilarityFunctionleftl(simfn,embeddings,leftop,rightop,numpy.max(synset2idx.values())+1,True)
    sol = SimilarityFunctionrell(simfn,embeddings,leftop,rightop,numpy.max(synset2idx.values())+1,True)

    posl = scipy.sparse.csr_matrix(cPickle.load(open(datpath+'XWN-lemme-lhs.pkl')),dtype='float32')
    posr = scipy.sparse.csr_matrix(cPickle.load(open(datpath+'XWN-lemme-rhs.pkl')),dtype='float32')
    poso = scipy.sparse.csr_matrix(cPickle.load(open(datpath+'XWN-lemme-rel.pkl')),dtype='float32')
    poslc = scipy.sparse.csr_matrix(cPickle.load(open(datpath+'XWN-corres-lhs.pkl')),dtype='float32')
    posrc = scipy.sparse.csr_matrix(cPickle.load(open(datpath+'XWN-corres-rhs.pkl')),dtype='float32')
    posoc = scipy.sparse.csr_matrix(cPickle.load(open(datpath+'XWN-corres-rel.pkl')),dtype='float32')

    nbtest = 5000

    llX , relX , rrX= calctestscore4(sll,srl,sol,posl[:,:nbtest],posr[:,:nbtest],poso[:,:nbtest],poslc[:,:nbtest],posrc[:,:nbtest],posoc[:,:nbtest])
    f = open(name +'/' + name + '_XWNrank.pkl','w')
    cPickle.dump(llX,f,-1)
    cPickle.dump(relX,f,-1)
    cPickle.dump(rrX,f,-1)

#----------------------------------------------------------------------------------------------------
#if evaln == 10 or evaln == 0:
#    srl = SimilarityFunctionrightl(simfn,embeddings,leftop,rightop,numpy.max(synset2idx.values())+1,True)
#    sll = SimilarityFunctionleftl(simfn,embeddings,leftop,rightop,numpy.max(synset2idx.values())+1,True)
#    sol = SimilarityFunctionrell(simfn,embeddings,leftop,rightop,numpy.max(synset2idx.values())+1,True)
#
#    posl = scipy.sparse.csr_matrix(cPickle.load(open(datpath+'XWN-lemme-lhs.pkl')),dtype='float32')
#    posr = scipy.sparse.csr_matrix(cPickle.load(open(datpath+'XWN-lemme-rhs.pkl')),dtype='float32')
#    poso = scipy.sparse.csr_matrix(cPickle.load(open(datpath+'XWN-lemme-rel.pkl')),dtype='float32')
#    poslc = scipy.sparse.csr_matrix(cPickle.load(open(datpath+'XWN-mod-lhs.pkl')),dtype='float32')
#    posrc = scipy.sparse.csr_matrix(cPickle.load(open(datpath+'XWN-mod-rhs.pkl')),dtype='float32')
#    posoc = scipy.sparse.csr_matrix(cPickle.load(open(datpath+'XWN-mod-rel.pkl')),dtype='float32')
#
#    nbtest = 5000
#
#    llX , relX , rrX= calctestscore4(sll,srl,sol,posl[:,:nbtest],posr[:,:nbtest],poso[:,:nbtest],poslc[:,:nbtest],posrc[:,:nbtest],posoc[:,:nbtest])
#    f = open(name +'/' + name + '_XWNmodrank.pkl','w')
#    cPickle.dump(llX,f,-1)
#    cPickle.dump(relX,f,-1)
#    cPickle.dump(rrX,f,-1)

#----------------------------------------------------------------------------------------------------
#if evaln == 11 or evaln == 0:
#    srl = SimilarityFunctionrightl(simfn,embeddings,leftop,rightop,numpy.max(synset2idx.values())+1,True)
#    sll = SimilarityFunctionleftl(simfn,embeddings,leftop,rightop,numpy.max(synset2idx.values())+1,True)
#    sol = SimilarityFunctionrell(simfn,embeddings,leftop,rightop,numpy.max(synset2idx.values())+1,True)
#
#    posl = scipy.sparse.csr_matrix(cPickle.load(open(datpath+'XWN-lemme-lhs.pkl')),dtype='float32')
#    posr = scipy.sparse.csr_matrix(cPickle.load(open(datpath+'XWN-lemme-rhs.pkl')),dtype='float32')
#    poso = scipy.sparse.csr_matrix(cPickle.load(open(datpath+'XWN-lemme-rel.pkl')),dtype='float32')
#    poslc = scipy.sparse.csr_matrix(cPickle.load(open(datpath+'XWN-nmod-lhs.pkl')),dtype='float32')
#    posrc = scipy.sparse.csr_matrix(cPickle.load(open(datpath+'XWN-nmod-rhs.pkl')),dtype='float32')
#    posoc = scipy.sparse.csr_matrix(cPickle.load(open(datpath+'XWN-nmod-rel.pkl')),dtype='float32')
#
#    nbtest = 5000
#
#    llX , relX , rrX= calctestscore4(sll,srl,sol,posl[:,:nbtest],posr[:,:nbtest],poso[:,:nbtest],poslc[:,:nbtest],posrc[:,:nbtest],posoc[:,:nbtest])
#    f = open(name +'/' + name + '_XWNnmodrank.pkl','w')
#    cPickle.dump(llX,f,-1)
#    cPickle.dump(relX,f,-1)
#    cPickle.dump(rrX,f,-1)


#----------------------------------------------------------------------------------------------------
if evaln == 2 or evaln == 0:
    modelpred = {}
    nmodelpred = {}
    posl = (cPickle.load(open(datpath+'XWN-WSD-lhs.pkl'))).tocsr()
    posr = (cPickle.load(open(datpath+'XWN-WSD-rhs.pkl'))).tocsr()
    poso = (cPickle.load(open(datpath+'XWN-WSD-rel.pkl'))).tocsr()
    dicto = cPickle.load(open(datpath+'XWN-WSD-dict.pkl'))
    lab = cPickle.load(open(datpath+'XWN-WSD-lab.pkl'))
    freq = cPickle.load(open(datpath+'XWN-WSD-freq.pkl'))
    simifunc = BatchSimilarityFunction(simfn,embeddings,leftop,rightop)
    listrank = (simifunc(posl,posr,poso)[0]).flatten()
    modelvX = []
    linvX = []
    softvX = []
    for idx,i in enumerate(dicto.keys()):
        listtmp = listrank[dicto[i][0]:dicto[i][1]]
        labtmp = lab[dicto[i][0]:dicto[i][1]]
        assert sum(labtmp)==1
        freqtmp = freq[dicto[i][0]:dicto[i][1]]
        lin_p = normafunc(listtmp)
        soft_p = softmaxfunc(listtmp)
        lintmp = lin_p * numpy.asarray(freqtmp)
        softtmp = soft_p *  numpy.asarray(freqtmp)
        if numpy.argsort(listtmp)[-1] != numpy.argsort(labtmp)[-1]:
            modelvX +=[1]
        else:
            modelvX +=[0]
        if numpy.argsort(lintmp)[-1] != numpy.argsort(labtmp)[-1]:
            linvX +=[1]
        else:
            linvX +=[0]
        if numpy.argsort(softtmp)[-1] != numpy.argsort(labtmp)[-1]:
            softvX +=[1]
        else:
            softvX += [0]
        modelpred.update({i:numpy.argsort(softtmp)[-1]})
        bbtr = True
        for zz in numpy.argsort(softtmp):
            if zz != numpy.argsort(softtmp)[-1] and zz != numpy.argsort(labtmp)[-1]:
                bbtr = False
                gt = zz
        if bbtr:
            nmodelpred.update({i:numpy.argsort(labtmp)[-1]})
        else:
            nmodelpred.update({i:gt})
    f = open(name +'/' + name + '_XWN-WSD.pkl','w')
    g = open(name +'/'+'modelpred.pkl','w')
    h = open(name +'/'+'nmodelpred.pkl','w')
    cPickle.dump(modelvX,f,-1)
    cPickle.dump(linvX,f,-1)
    cPickle.dump(softvX,f,-1)
    cPickle.dump(modelpred,g,-1)
    cPickle.dump(nmodelpred,h,-1)
    h.close()
    f.close()
    g.close()

#----------------------------------------------------------------------------------------------------
if evaln == 3 or evaln == 0:
    srl = SimilarityFunctionrightl(simfn,embeddings,leftop,rightop,numpy.max(synset2idx.values())+1,True)
    sll = SimilarityFunctionleftl(simfn,embeddings,leftop,rightop,numpy.max(synset2idx.values())+1,True)
    sol = SimilarityFunctionrell(simfn,embeddings,leftop,rightop,numpy.max(synset2idx.values())+1,True)
    posl = scipy.sparse.csr_matrix(cPickle.load(open(datpath+'Brown-lemme-lhs.pkl')),dtype='float32')
    posr = scipy.sparse.csr_matrix(cPickle.load(open(datpath+'Brown-lemme-rhs.pkl')),dtype='float32')
    poso = scipy.sparse.csr_matrix(cPickle.load(open(datpath+'Brown-lemme-rel.pkl')),dtype='float32')
    poslc = scipy.sparse.csr_matrix(cPickle.load(open(datpath+'Brown-corres-lhs.pkl')),dtype='float32')
    posrc = scipy.sparse.csr_matrix(cPickle.load(open(datpath+'Brown-corres-rhs.pkl')),dtype='float32')
    posoc = scipy.sparse.csr_matrix(cPickle.load(open(datpath+'Brown-corres-rel.pkl')),dtype='float32')

    nbtest = 5000

    llB , relB , rrB= calctestscore4(sll,srl,sol,posl[:,:nbtest],posr[:,:nbtest],poso[:,:nbtest],poslc[:,:nbtest],posrc[:,:nbtest],posoc[:,:nbtest])
    f = open(name +'/' + name + '_Brownrank.pkl','w')
    cPickle.dump(llB,f,-1)
    cPickle.dump(relB,f,-1)
    cPickle.dump(rrB,f,-1)


#----------------------------------------------------------------------------------------------------
if evaln == 4 or evaln == 0:
    posl = (cPickle.load(open(datpath+'Brown-WSD-lhs.pkl'))).tocsr()
    posr = (cPickle.load(open(datpath+'Brown-WSD-rhs.pkl'))).tocsr()
    poso = (cPickle.load(open(datpath+'Brown-WSD-rel.pkl'))).tocsr()
    dicto = cPickle.load(open(datpath+'Brown-WSD-dict.pkl'))
    lab = cPickle.load(open(datpath+'Brown-WSD-lab.pkl'))
    freq = cPickle.load(open(datpath+'Brown-WSD-freq.pkl'))
    simifunc = BatchSimilarityFunction(simfn,embeddings,leftop,rightop)
    listrank = (simifunc(posl,posr,poso)[0]).flatten()
    modelvB = []
    linvB = []
    softvB = []
    for idx,i in enumerate(dicto.keys()):
        listtmp = listrank[dicto[i][0]:dicto[i][1]]
        labtmp = lab[dicto[i][0]:dicto[i][1]]
        assert sum(labtmp)==1
        freqtmp = freq[dicto[i][0]:dicto[i][1]]
        lin_p = normafunc(listtmp)
        soft_p = softmaxfunc(listtmp)
        lintmp = lin_p * numpy.asarray(freqtmp)
        softtmp = soft_p *  numpy.asarray(freqtmp)
        if numpy.argsort(listtmp)[-1] != numpy.argsort(labtmp)[-1]:
            modelvB +=[1]
        else:
            modelvB +=[0]
        if numpy.argsort(lintmp)[-1] != numpy.argsort(labtmp)[-1]:
            linvB +=[1]
        else:
            linvB +=[0]
        if numpy.argsort(softtmp)[-1] != numpy.argsort(labtmp)[-1]:
            softvB +=[1]
        else:
            softvB += [0]
    f = open(name +'/' + name + '_Brown-WSD.pkl','w')
    cPickle.dump(modelvB,f,-1)
    cPickle.dump(linvB,f,-1)
    cPickle.dump(softvB,f,-1)

#----------------------------------------------------------------------------------------------------
if evaln == 5 or evaln == 0:
    posl = (cPickle.load(open(datpath+'Senseval3-WSD-lhs.pkl'))).tocsr()
    posr = (cPickle.load(open(datpath+'Senseval3-WSD-rhs.pkl'))).tocsr()
    poso = (cPickle.load(open(datpath+'Senseval3-WSD-rel.pkl'))).tocsr()
    dicto = cPickle.load(open(datpath+'Senseval3-WSD-dict.pkl'))
    lab = cPickle.load(open(datpath+'Senseval3-WSD-lab.pkl'))
    freq = cPickle.load(open(datpath+'Senseval3-WSD-freq.pkl'))
    simifunc = BatchSimilarityFunction(simfn,embeddings,leftop,rightop)
    listrank = (simifunc(posl,posr,poso)[0]).flatten()
    modelvX = []
    linvX = []
    softvX = []
    for idx,i in enumerate(dicto.keys()):
        listtmp = listrank[dicto[i][0]:dicto[i][1]]
        labtmp = lab[dicto[i][0]:dicto[i][1]]
        assert sum(labtmp)==1
        freqtmp = freq[dicto[i][0]:dicto[i][1]]
        lin_p = normafunc(listtmp)
        soft_p = softmaxfunc(listtmp)
        lintmp = lin_p * numpy.asarray(freqtmp)
        softtmp = soft_p *  numpy.asarray(freqtmp)
        if numpy.argsort(listtmp)[-1] != numpy.argsort(labtmp)[-1]:
            modelvX +=[1]
        else:
            modelvX +=[0]
        if numpy.argsort(lintmp)[-1] != numpy.argsort(labtmp)[-1]:
            linvX +=[1]
        else:
            linvX +=[0]
        if numpy.argsort(softtmp)[-1] != numpy.argsort(labtmp)[-1]:
            softvX +=[1]
        else:
            softvX += [0]
    f = open(name +'/' + name + '_Senseval3-WSD.pkl','w')
    cPickle.dump(modelvX,f,-1)
    cPickle.dump(linvX,f,-1)
    cPickle.dump(softvX,f,-1)


#----------------------------------------------------------------------------------------------------
if evaln == 6 or evaln == 0:
    datpath = ''

    # valid set
    WNvall = (cPickle.load(open(datpath+'WordNet3.0-val-lhs.pkl','r'))).tocsr()
    WNvalr = (cPickle.load(open(datpath+'WordNet3.0-val-rhs.pkl','r'))).tocsr()
    WNvalo = (cPickle.load(open(datpath+'WordNet3.0-val-rel.pkl','r'))).tocsr()

    # test set
    WNtestl = (cPickle.load(open(datpath+'WordNet3.0-test-lhs.pkl','r'))).tocsr()
    WNtestr = (cPickle.load(open(datpath+'WordNet3.0-test-rhs.pkl','r'))).tocsr()
    WNtesto = (cPickle.load(open(datpath+'WordNet3.0-test-rel.pkl','r'))).tocsr()

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

    sl = SimilarityFunctionleft(simfn,embeddings,leftop,rightop,subtensorspec = numpy.max(synset2idx.values())+1)
    sr = SimilarityFunctionright(simfn,embeddings,leftop,rightop,subtensorspec = numpy.max(synset2idx.values())+1)

    errlval,errrval = calctestval2(sl,sr,idxtl,idxtr,idxto)
    errltes,errrtes = calctestval2(sl,sr,idxvl,idxvr,idxvo)

    f = open(name +'/' + name + '_WN-rank.pkl','w')
    cPickle.dump(errlval,f,-1)
    cPickle.dump(errrval,f,-1)
    cPickle.dump(errltes,f,-1)
    cPickle.dump(errrtes,f,-1)



