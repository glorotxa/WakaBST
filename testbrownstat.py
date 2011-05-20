import scipy.sparse
import cPickle
import os
import sys
from model import *

def reorder(listl,ordering):
    newll = []
    for i in ordering:
        newll+=[listl[i]]
    return newll

def parseconcept(concept):
    if concept[:2]=='__':
        concept =  concept.split('_')
        namecc = concept[2]
        for tmp in concept[3:-1]:
            namecc += '_'+tmp
        return namecc, concept[-1]
    else:
        return None

def parseline(line):
    lhs,rel,rhs = line.split('\t')
    lhs = lhs.split(' ')
    rhs = rhs.split(' ')
    return lhs,rel,rhs

def normafunc(x):
    return (x-numpy.min(x))/sum((x-numpy.min(x)))

def softmaxfunc(x):
    return numpy.exp(x-numpy.max(x))/sum(numpy.exp(x-numpy.max(x)))

def results(x,listtuples,Eb):
    ordered = numpy.sort(x)
    bestscore = ordered[-1]
    ctsame = 1
    while (len(x)-ctsame!=0) and ordered[-1-ctsame] == bestscore:
        ctsame+=1
    ctsame = numpy.random.randint(ctsame)
    ordering = numpy.argsort(x)
    #
    firstscore = ordering[-1-ctsame]
    tupleselected = listtuples[firstscore]
    goodtuple = listtuples[-1]
    err_ind = 0
    ct = 0
    for i,j in zip(tupleselected,goodtuple):
        ct+=1
        if i!=j:
            err_ind+=1
    #print 'err_ind',err_ind,ct
    if tupleselected != goodtuple:
        err_tot = 1
    else:
        err_tot = 0
    #print 'err_tot',err_tot
    #
    rank_tot = (len(x)-(numpy.argsort((numpy.sort(x) == x[-1]))[-1]+1))/float(len(x))
    #print 'rank_tot',rank_tot
    rank_ind = 0
    for i in range(len(goodtuple)):
        ll = []
        currentset = set([tupleselected[i]])
        tmpcnt = 0
        while goodtuple[i] not in list(currentset):
            currentset = set(list(currentset)+[listtuples[-1-tmpcnt][i]])
            tmpcnt += 1
        for tt in listtuples:
            ll+=tt[i]
        if len(set(ll))!=1:
            rank_ind += (len(currentset)-1) / float(len(set(ll))-1)
    #
    #print 'rank_ind',rank_ind
    d_ind = 0
    for i,j in zip(tupleselected,goodtuple):
        d_ind += numpy.sqrt(numpy.sum((Eb.E.value[:,synset2idx[i]] - Eb.E.value[:,synset2idx[j]])**2))
    #print 'd_ind',d_ind
    return [ct,err_ind,err_tot,rank_ind,rank_tot,d_ind]
    
    

#for i in dictconcept.keys():
#    freq = 0
#    for j in dictconcept[i]:
#        freq += concept2freq['__'+i+'_'+j]
#    if abs(freq - 1.0)>10E-10:
#        print freq 
#        cool=True

synset2lemme = cPickle.load(open('synset2lemme.pkl','r'))
lemme2synset = cPickle.load(open('lemme2synset.pkl','r'))
lemme2freq = cPickle.load(open('lemme2freq.pkl','r'))
synset2idx = cPickle.load(open('synset2idx.pkl','r'))
idx2synset = cPickle.load(open('idx2synset.pkl','r'))
lemme2idx = cPickle.load(open('lemme2idx.pkl','r'))
idx2lemme = cPickle.load(open('idx2lemme.pkl','r'))
synset2neg = cPickle.load(open('synset2neg.pkl','r'))
synset2def = cPickle.load(open('synset2def.pkl','r'))
synset2concept = cPickle.load(open('synset2concept.pkl','r'))


loadmodel = 'expe50/model.pkl'

f = open(loadmodel)
embeddings = cPickle.load(f)
leftop = cPickle.load(f)
rightop = cPickle.load(f)
simfn = eval('dotsim')

simifunc = BatchSimilarityFunction(simfn,embeddings,leftop,rightop)

leftopid = Id()
rightopid = Id()
Esim = BatchSimilarityFunction(L2sim,embeddings,leftopid,rightopid)


f = open('/data/lisa/data/NLU/semcor3.0/brown-synsets/Brown-filtered-triplets-unambiguous-lemmas.dat','r')
g = open('/data/lisa/data/NLU/semcor3.0/brown-synsets/Brown-filtered-triplets-unambiguous-synsets.dat','r')

dat1 = f.readlines()
f.close()
dat2 = g.readlines()
g.close()

dictres={}

dictres['rand']=numpy.zeros((5,))
dictres['freq']=numpy.zeros((5,))
dictres['L2_lr']=numpy.zeros((5,))
dictres['nf_L2_lr']=numpy.zeros((5,))
dictres['sf_L2_lr']=numpy.zeros((5,))
dictres['L2_tot']=numpy.zeros((5,))
dictres['nf_L2_tot']=numpy.zeros((5,))
dictres['sf_L2_tot']=numpy.zeros((5,))
dictres['model']=numpy.zeros((5,))
dictres['nf_model']=numpy.zeros((5,))
dictres['sf_model']=numpy.zeros((5,))

count_ind=0
count_tot=0
for i,k in zip(dat1,dat2):
    lhs,rel,rhs = parseline(i[:-1])
    lhsr,relr,rhsr = parseline(k[:-1])
    ct = 1
    for j in lhs:
        ct *= len(lemme2synset[j])
    j = rel
    ct *= len(lemme2synset[j])
    for j in rhs:
        ct *= len(lemme2synset[j])
    
    if ct<10000:
        posl = scipy.sparse.lil_matrix((embeddings.E.value.shape[1],ct),dtype=theano.config.floatX)
        posr = scipy.sparse.lil_matrix((embeddings.E.value.shape[1],ct),dtype=theano.config.floatX)
        poso = scipy.sparse.lil_matrix((embeddings.E.value.shape[1],ct),dtype=theano.config.floatX)

        currenttuple = [()]
        listfreq = [[]]
        for j,l in zip(lhs,lhsr):
            listconceptcurr = lemme2synset[j]
            tmpfreq = lemme2freq[j]
            idxs = listconceptcurr.index(l)
            oldval = tmpfreq.pop(idxs)
            tmpfreq += [oldval]
            listconceptcurr.remove(l)
            listconceptcurr += [l]
            currentlist = []
            currentfreq = []
            for m,o in zip(currenttuple,listfreq):
                for n,p in zip(listconceptcurr,tmpfreq):
                    currentlist += [m+(n,)]
                    currentfreq += [o+[p,]]
            listfreq = currentfreq
            currenttuple = currentlist

        j=rel
        l=relr
        listconceptcurr = lemme2synset[j]
        tmpfreq = lemme2freq[j]
        idxs = listconceptcurr.index(l)
        oldval = tmpfreq.pop(idxs)
        tmpfreq += [oldval]
        listconceptcurr.remove(l)
        listconceptcurr += [l]
        currentlist = []
        currentfreq = []
        for m,o in zip(currenttuple,listfreq):
            for n,p in zip(listconceptcurr,tmpfreq):
                currentlist += [m+(n,)]
                currentfreq += [o+[p,]]
        listfreq = currentfreq
        currenttuple = currentlist

        for j,l in zip(rhs,rhsr):
            listconceptcurr = lemme2synset[j] 
            tmpfreq = lemme2freq[j]
            idxs = listconceptcurr.index(l)
            oldval = tmpfreq.pop(idxs)
            tmpfreq += [oldval]
            listconceptcurr.remove(l)
            listconceptcurr += [l]
            currentlist = []
            currentfreq = []
            for m,o in zip(currenttuple,listfreq):
                for n,p in zip(listconceptcurr,tmpfreq):
                    currentlist += [m+(n,)]
                    currentfreq += [o+[p,]]
            listfreq = currentfreq
            currenttuple = currentlist
        
        newlistfreq = []
        for j in listfreq:
            newlistfreq+=[numpy.prod(j)]
        listfreq = newlistfreq
        
        for idx,j in enumerate(currenttuple):
            ct = 0
            for k in xrange(ct,ct+len(lhs)):
                posl[synset2idx[j[k]],idx] += 1 / float(len(lhs))
            ct += len(lhs)
            poso[synset2idx[j[ct]],idx] = 1
            ct += 1
            for k in xrange(ct,ct+len(rhs)):
                posr[synset2idx[j[k]],idx] += 1 / float(len(rhs))
        import copy
        listsimi = simifunc(posl,posr,poso)[0]
        listrandom = numpy.random.permutation(len(listsimi))
        listL2hs = Esim(posl,posr,poso)[0]
        listL2tot = copy.copy(listL2hs)
        listL2tot += Esim(posl,poso,poso)[0]
        listL2tot += Esim(poso,posr,poso)[0]
        #print 'random-------------------------------------------------------------'
        tmpres = results(listrandom,currenttuple,embeddings)
        count_ind += tmpres[0]
        count_tot += 1
        dictres['rand']+=numpy.asarray(tmpres[1:])
        #print 'freq-------------------------------------------------------------'
        tmpres = results(numpy.asarray(listfreq),currenttuple,embeddings)
        dictres['freq']+=numpy.asarray(tmpres[1:])
        #print 'L2lr-------------------------------------------------------------'
        tmpres = results(listL2hs,currenttuple,embeddings)
        dictres['L2_lr']+= numpy.asarray(tmpres[1:])
        #print 'nfL2lr-------------------------------------------------------------'
        tmpres = results(normafunc(listL2hs)*numpy.asarray(listfreq),currenttuple,embeddings)
        dictres['nf_L2_lr']+=numpy.asarray(tmpres[1:])
        #print 'nsL2lr-------------------------------------------------------------'
        tmpres = results(softmaxfunc(listL2hs)*numpy.asarray(listfreq),currenttuple,embeddings)
        dictres['sf_L2_lr']+=numpy.asarray(tmpres[1:])
        #print 'l2tot-------------------------------------------------------------'
        tmpres = results(listL2tot,currenttuple,embeddings)
        dictres['L2_tot']+=numpy.asarray(tmpres[1:])
        #print 'nfL2tot-------------------------------------------------------------'
        tmpres = results(normafunc(listL2tot)*numpy.asarray(listfreq),currenttuple,embeddings)
        dictres['nf_L2_tot']+=numpy.asarray(tmpres[1:])
        #print 'sfL2tot-------------------------------------------------------------'
        tmpres = results(softmaxfunc(listL2tot)*numpy.asarray(listfreq),currenttuple,embeddings)
        dictres['sf_L2_tot']+=numpy.asarray(tmpres[1:])
        #print 'model-------------------------------------------------------------'
        tmpres = results(listsimi,currenttuple,embeddings)
        dictres['model']+=numpy.asarray(tmpres[1:])
        #print 'nfmodel-------------------------------------------------------------'
        tmpres = results(normafunc(listsimi)*numpy.asarray(listfreq),currenttuple,embeddings)
        dictres['nf_model']+=numpy.asarray(tmpres[1:])
        #print 'sfmodel-------------------------------------------------------------'
        tmpres = results(softmaxfunc(listsimi)*numpy.asarray(listfreq),currenttuple,embeddings)
        dictres['sf_model']+=numpy.asarray(tmpres[1:])
        lhs,rel,rhs = parseline(i[:-1])
        print ''
        print ' ------------------ SAMPLE:',count_tot
        for i in ['rand','freq','L2_lr','nf_L2_lr','sf_L2_lr','L2_tot','nf_L2_tot','sf_L2_tot','model','nf_model','sf_model']:
            print '#################'
            print i
            print 'err_ind:', dictres[i][0]/float(count_ind),'err_tot:', dictres[i][1]/float(count_tot),'rank_ind:', dictres[i][2]/float(count_ind),'rank_tot:', dictres[i][3]/float(count_tot),'rank_ind_N:', dictres[i][2]/float(dictres[i][0]),'rank_tot_N:', dictres[i][3]/float(dictres[i][1]),'d_ind:', dictres[i][4]/float(count_ind),'d_ind_N:', dictres[i][4]/float(dictres[i][0])
