import scipy.sparse
import cPickle
import os
import sys
from model import *


def normafunc(x):
    return (x-numpy.min(x))/sum((x-numpy.min(x)))

def softmaxfunc(x):
    return numpy.exp(x-numpy.max(x))/sum(numpy.exp(x-numpy.max(x)))

loadmodel = '/mnt/scratch/bengio/glorotxa/data/exp/glorotxa_db/wakabstfinal/98/model.pkl'

f = open(loadmodel)
embeddings = cPickle.load(f)
leftop = cPickle.load(f)
rightop = cPickle.load(f)
simfn = eval('dotsim')

#MLPout = cPickle.load(f)
#simfn = MLPout

simifunc = BatchSimilarityFunction(simfn,embeddings,leftop,rightop)

posl = cPickle.load(open('/mnt/scratch/bengio/glorotxa/data/exp/WakaBST3/Brown-WSD-lhs.pkl'))
posr = cPickle.load(open('/mnt/scratch/bengio/glorotxa/data/exp/WakaBST3/Brown-WSD-rhs.pkl'))
poso = cPickle.load(open('/mnt/scratch/bengio/glorotxa/data/exp/WakaBST3/Brown-WSD-rel.pkl'))
dicto = cPickle.load(open('/mnt/scratch/bengio/glorotxa/data/exp/WakaBST3/Brown-WSD-dict.pkl'))
lab = cPickle.load(open('/mnt/scratch/bengio/glorotxa/data/exp/WakaBST3/Brown-WSD-lab.pkl'))
freq = cPickle.load(open('/mnt/scratch/bengio/glorotxa/data/exp/WakaBST3/Brown-WSD-freq.pkl'))

listrank = simifunc(posl,posr,poso)[0]

randv = 0
freqv = 0
modelv = 0
linv = 0
softv = 0
addlv = 0
addsoftv = 0


randvs = 0
freqvs = 0
modelvs = 0
linvs = 0
softvs = 0
addlvs = 0
addsoftvs = 0

ct = 0


randvn = 0
freqvn = 0
modelvn = 0
linvn = 0
softvn = 0
addlvn = 0
addsoftvn = 0

ctn = 0


partition = 100.

for idx,i in enumerate(dicto.keys()):
    listtmp = listrank[dicto[i][0]:dicto[i][1]]
    labtmp = lab[dicto[i][0]:dicto[i][1]]
    assert sum(labtmp)==1
    freqtmp = freq[dicto[i][0]:dicto[i][1]]
    lin_p = normafunc(listtmp)
    soft_p = normafunc(listtmp)
    lintmp = lin_p * numpy.asarray(freqtmp)
    softtmp = soft_p *  numpy.asarray(freqtmp)
    addltmp = lin_p + partition * numpy.asarray(freqtmp)
    addsofttmp = soft_p +  partition * numpy.asarray(freqtmp)
    if numpy.argsort(listtmp)[-1] != numpy.argsort(labtmp)[-1]:
        modelv +=1
    if numpy.argsort(lintmp)[-1] != numpy.argsort(labtmp)[-1]:
        linv +=1
    if numpy.argsort(softtmp)[-1] != numpy.argsort(labtmp)[-1]:
        softv +=1
    if numpy.argsort(addltmp)[-1] != numpy.argsort(labtmp)[-1]:
        addlv +=1
    if numpy.argsort(addsofttmp)[-1] != numpy.argsort(labtmp)[-1]:
        addsoftv +=1
    if numpy.argsort(freqtmp)[-1] != numpy.argsort(labtmp)[-1]:
        freqv +=1
    randv += (len(listtmp)-1)/float(len(listtmp))
    if dicto[i][2]:
        if numpy.argsort(listtmp)[-1] != numpy.argsort(labtmp)[-1]:
            modelvs +=1
        if numpy.argsort(lintmp)[-1] != numpy.argsort(labtmp)[-1]:
            linvs +=1
        if numpy.argsort(softtmp)[-1] != numpy.argsort(labtmp)[-1]:
            softvs +=1
        if numpy.argsort(addltmp)[-1] != numpy.argsort(labtmp)[-1]:
            addlvs +=1
        if numpy.argsort(addsofttmp)[-1] != numpy.argsort(labtmp)[-1]:
            addsoftvs +=1
        if numpy.argsort(freqtmp)[-1] != numpy.argsort(labtmp)[-1]:
            freqvs +=1
        randvs += (len(listtmp)-1)/float(len(listtmp))
        ct += 1
    else:
        if numpy.argsort(listtmp)[-1] != numpy.argsort(labtmp)[-1]:
            modelvn +=1
        if numpy.argsort(lintmp)[-1] != numpy.argsort(labtmp)[-1]:
            linvn +=1
        if numpy.argsort(softtmp)[-1] != numpy.argsort(labtmp)[-1]:
            softvn +=1
        if numpy.argsort(addltmp)[-1] != numpy.argsort(labtmp)[-1]:
            addlvn +=1
        if numpy.argsort(addsofttmp)[-1] != numpy.argsort(labtmp)[-1]:
            addsoftvn +=1
        if numpy.argsort(freqtmp)[-1] != numpy.argsort(labtmp)[-1]:
            freqvn +=1
        randvn += (len(listtmp)-1)/float(len(listtmp))
        ctn+=1
    print randv/float(idx+1),modelv/float(idx+1),freqv/float(idx+1),linv/float(idx+1),softv/float(idx+1),addlv/float(idx+1),addsoftv/float(idx+1)

print randvs/float(ct),modelvs/float(ct),freqvs/float(ct),linvs/float(ct),softvs/float(ct),addlvs/float(ct),addsoftvs/float(ct)
print randvn/float(ctn),modelvn/float(ctn),freqvn/float(ctn),linvn/float(ctn),softvn/float(ctn),addlvn/float(ctn),addsoftvn/float(ctn)
# unambiguous instance:
idx = idx+1+7737

print randv/float(idx+1),modelv/float(idx+1),freqv/float(idx+1),linv/float(idx+1),softv/float(idx+1),addlv/float(idx+1),addsoftv/float(idx+1)
