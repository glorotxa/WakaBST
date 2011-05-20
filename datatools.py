import numpy

f = open('/data/lisa/data/NLU/wordnet3.0-synsets/filtered-data/WordNet3.0-filtered-lemma2synsets+cnts.txt','r')

dat = f.readlines()
f.close()
dat = dat



lemme2synset = {}
lemme2freq = {}

for idx,i in enumerate(dat):
    lemme,synsets,frequence = i[:-1].split('\t')
    if synsets[0] != '_':
        synsets = synsets[:-1]
        frequence = frequence[:-1]
    synlist = synsets.split(' ')
    freqlist = list(numpy.asarray(frequence.split(' '),dtype='float64'))
    lemme2synset.update({lemme:synlist})
    lemme2freq.update({lemme:freqlist})


import cPickle

f = open('lemme2synset.pkl','w')
g = open('lemme2freq.pkl','w')

cPickle.dump(lemme2synset,f,-1)
cPickle.dump(lemme2freq,g,-1)

f.close()
g.close()

f = open('/data/lisa/data/NLU/wordnet3.0-synsets/filtered-data/WordNet3.0-filtered-synset2lemmas.txt','r')
dat = f.readlines()
f.close()

synset2lemme = {}
synset2idx = {}
idx2synset = {}

for idx,i in enumerate(dat):
    synset,lemmes = i[:-1].split('\t')
    lemmes=lemmes[:-1]
    lemmelist = lemmes.split(' ')
    synset2lemme.update({synset:lemmelist})
    synset2idx.update({synset:idx})
    idx2synset.update({idx:synset})

synsetnb = idx+1

for j in lemme2synset.keys():
    if j[1]!='_':
        synset2lemme.update({j:[j]})
        synset2idx.update({j:synsetnb})
        idx2synset.update({synsetnb:j})
        synsetnb+=1

f = open('synset2lemme.pkl','w')
g = open('synset2idx.pkl','w')
h = open('idx2synset.pkl','w')


cPickle.dump(synset2lemme,f,-1)
cPickle.dump(synset2idx,g,-1)
cPickle.dump(idx2synset,h,-1)

f.close()
g.close()
h.close()

f = open('/data/lisa/data/NLU/wordnet3.0-synsets/filtered-data/WordNet3.0-filtered-synset2definitions.txt','r')
dat = f.readlines()
f.close()

synset2def = {}
synset2concept = {}

for idx,i in enumerate(dat):
    synset,concept,definition = i[:-1].split('\t')
    synset2def.update({synset:definition})
    synset2concept.update({synset:concept})


f = open('synset2def.pkl','w')
g = open('synset2concept.pkl','w')

cPickle.dump(synset2def,f,-1)
cPickle.dump(synset2concept,g,-1)

f.close()
g.close()

f = open('/data/lisa/data/NLU/wordnet3.0-synsets/filtered-data/WordNet3.0-filtered-synset2negative_synsets.txt','r')
dat = f.readlines()
f.close()

synset2neg = {}

for idx,i in enumerate(dat):
    synset,neg = i[:-1].split('\t')
    neg = neg[:-1]
    synset2neg.update({synset:neg.split(' ')})

f = open('synset2neg.pkl','w')

cPickle.dump(synset2neg,f,-1)

f.close()


f = open('/data/lisa/data/NLU/wordnet3.0-synsets/filtered-data/WordNet3.0-filtered-oldname2synset.txt','r')
dat = f.readlines()
f.close()

concept2synset = {}

for idx,i in enumerate(dat):
    concept,synset= i[:-1].split('\t')
    concept2synset.update({concept:synset})

f = open('concept2synset.pkl','w')

cPickle.dump(concept2synset,f,-1)

f.close()


lemme2idx = {}
idx2lemme = {}

ct = synsetnb
for i in lemme2synset.keys():
    if len(lemme2synset[i])>1:
        lemme2idx.update({i:ct})
        idx2lemme.update({ct:i})
        ct+=1
    else:
        lemme2idx.update({i:synset2idx[lemme2synset[i][0]]})
   
f = open('lemme2idx.pkl','w')
g = open('idx2lemme.pkl','w')

cPickle.dump(lemme2idx,f,-1)
cPickle.dump(idx2lemme,g,-1)

f.close()
g.close()


def parseline(line):
    lhs,rel,rhs = line.split('\t')
    lhs = lhs.split(' ')
    rhs = rhs.split(' ')
    rel = rel.split(' ')
    return lhs,rel,rhs

if False:
    speciallist = ['_substance_holonym','_attribute','_substance_meronym','_entailment','_cause']
    for datatyp in ['train','val','test']:
        for nolemme in [True,False]:
            f = open('/data/lisa/data/NLU/wordnet3.0-synsets/filtered-data/%s-WordNet3.0-filtered-synsets-relations-anto.txt'%datatyp,'r')

            dat = f.readlines()
            f.close()

            ct = 0
            for i in dat:
                lhs,rel,rhs = parseline(i[:-1])
                if rel[0] not in speciallist:
                    ct += 1
                    if not nolemme:
                        for j in synset2lemme[lhs[0]]:
                            if len(lemme2synset[j])!=1:
                                ct += 1
                        for j in synset2lemme[rel[0]]:
                            if len(lemme2synset[j])!=1:
                                ct += 1
                        for j in synset2lemme[rhs[0]]:
                            if len(lemme2synset[j])!=1:
                                ct += 1

            print len(dat),ct

            import scipy.sparse
            if not nolemme or datatyp == 'train':
                posl = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,ct),dtype='float32')
                posr = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,ct),dtype='float32')
                poso = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,ct),dtype='float32')
            else:    
                posl = scipy.sparse.lil_matrix((numpy.max(synset2idx.values())+1,ct),dtype='float32')
                posr = scipy.sparse.lil_matrix((numpy.max(synset2idx.values())+1,ct),dtype='float32')
                poso = scipy.sparse.lil_matrix((numpy.max(synset2idx.values())+1,ct),dtype='float32')
            ct = 0
            for i in dat:
                lhs,rel,rhs = parseline(i[:-1])
                if rel[0] not in speciallist:
                    posl[synset2idx[lhs[0]],ct]=1
                    posr[synset2idx[rhs[0]],ct]=1
                    poso[synset2idx[rel[0]],ct]=1
                    ct+=1
                    if not nolemme:
                        for j in synset2lemme[lhs[0]]:
                            if len(lemme2synset[j])!=1: 
                                posl[lemme2idx[j],ct]=1
                                posr[synset2idx[rhs[0]],ct]=1
                                poso[synset2idx[rel[0]],ct]=1
                                ct += 1
                        for j in synset2lemme[rel[0]]:
                            if len(lemme2synset[j])!=1:
                                posl[synset2idx[lhs[0]],ct]=1
                                posr[synset2idx[rhs[0]],ct]=1
                                poso[lemme2idx[j],ct]=1
                                ct += 1
                        for j in synset2lemme[rhs[0]]:
                            if len(lemme2synset[j])!=1:
                                posr[lemme2idx[j],ct]=1
                                posl[synset2idx[lhs[0]],ct]=1
                                poso[synset2idx[rel[0]],ct]=1
                                ct += 1
            
            if not nolemme:
                f = open('WordNet3.0-%s-lhs.pkl'%datatyp,'w')
                g = open('WordNet3.0-%s-rhs.pkl'%datatyp,'w')
                h = open('WordNet3.0-%s-rel.pkl'%datatyp,'w')
            else:
                f = open('WordNet3.0-easy-%s-lhs.pkl'%datatyp,'w')
                g = open('WordNet3.0-easy-%s-rhs.pkl'%datatyp,'w')
                h = open('WordNet3.0-easy-%s-rel.pkl'%datatyp,'w')
                
            cPickle.dump(posl.tocsr(),f,-1)
            cPickle.dump(posr.tocsr(),g,-1)
            cPickle.dump(poso.tocsr(),h,-1)

            f.close()
            g.close()
            h.close()

if False:
    totalsize = 0
    for nbf in range(131):
        f = open('/data/lisa/data/NLU/converted-wikipedia/nlu-data/triplets-file-%s.dat'%nbf,'r')
        dat = f.readlines()
        f.close()
        totalsize+=len(dat)
    f = open('/data/lisa/data/NLU/converted-wikipedia/nlu-data-synsets/unambiguous-triplets.dat','r')
    dat = f.readlines()
    totalsize+=len(dat)

    import scipy.sparse
    posl = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,totalsize),dtype='float32')
    posr = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,totalsize),dtype='float32')
    poso = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,totalsize),dtype='float32')


    currentidx = 0

    for nbf in range(131):
        f = open('/data/lisa/data/NLU/converted-wikipedia/nlu-data/triplets-file-%s.dat'%nbf,'r')
        dat = f.readlines()
        print 'triplets-file-%s.dat'%nbf
        for i in dat:
            lhs,rel,rhs = parseline(i[:-1])
            if lhs[-1]=='':
                lhs = lhs[:-1]
            if rhs[-1]=='':
                rhs = rhs[:-1]
            if rel[-1]=='':
                rel = rel[:-1]
            for j in lhs:
                posl[lemme2idx[j],currentidx]+=1/float(len(lhs))
            for j in rhs:
                posr[lemme2idx[j],currentidx]+=1/float(len(rhs))
            for j in rel:
                poso[lemme2idx[j],currentidx]+=1/float(len(rel))
            currentidx += 1 


    f = open('/data/lisa/data/NLU/converted-wikipedia/nlu-data-synsets/unambiguous-triplets.dat','r')
    dat = f.readlines()
    print len(dat)
    for i in dat:
        lhs,rel,rhs = parseline(i[:-1])
        if lhs[-1]=='':
            lhs = lhs[:-1]
        if rhs[-1]=='':
            rhs = rhs[:-1]
        if rel[-1]=='':
            rel = rel[:-1]
        for j in lhs:
            posl[synset2idx[j],currentidx]+=1/float(len(lhs))
        for j in rhs:
            posr[synset2idx[j],currentidx]+=1/float(len(rhs))
        for j in rel:
            poso[synset2idx[j],currentidx]+=1/float(len(rel))
        currentidx += 1

    assert currentidx == totalsize
    print "finished"
    neworder = numpy.random.permutation(totalsize)

    poso = (poso.tocsr())[:,neworder]
    posl = (posl.tocsr())[:,neworder]
    posr = (posr.tocsr())[:,neworder]

    f = open('Wikilemmes-lhs.pkl','w')
    g = open('Wikilemmes-rhs.pkl','w')
    h = open('Wikilemmes-rel.pkl','w')

    cPickle.dump(posl,f,-1)
    cPickle.dump(posr,g,-1)
    cPickle.dump(poso,h,-1)

    f.close()
    g.close()
    h.close()




    f = open('/data/lisa/data/NLU/converted-wikipedia/nlu-data-synsets/lessambiguous-triplets.dat','r')
    dat = f.readlines()

    ct = 0

    for i in dat:
        onlyonesynset = True
        lhs,rel,rhs = parseline(i[:-1])
        if lhs[-1]=='':
            lhs = lhs[:-1]
        if rhs[-1]=='':
            rhs = rhs[:-1]
        if rel[-1]=='':
            rel = rel[:-1]
        for j in lhs:
            if j[0]!='_':
                assert onlyonesynset
                onlyonesynset = False
                ct+=len(synset2neg[j])
        for j in rhs:
            if j[0]!='_':
                assert onlyonesynset
                onlyonesynset = False
                ct+=len(synset2neg[j])
        for j in rel:
            if j[0]!='_':
                assert onlyonesynset
                onlyonesynset = False
                ct+=len(synset2neg[j])
                
    import scipy.sparse

    posln = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,ct),dtype='float32')
    posrn = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,ct),dtype='float32')
    poson = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,ct),dtype='float32')
    posl = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,ct),dtype='float32')
    posr = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,ct),dtype='float32')
    poso = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,ct),dtype='float32')
    print len(dat),ct

    f = open('/data/lisa/data/NLU/converted-wikipedia/nlu-data-synsets/lessambiguous-triplets.dat','r')
    dat = f.readlines()


    currentidx = 0

    for i in dat:
        onlyonesynset = True
        lhs,rel,rhs = parseline(i[:-1])
        if lhs[-1]=='':
            lhs = lhs[:-1]
        if rhs[-1]=='':
            rhs = rhs[:-1]
        if rel[-1]=='':
            rel = rel[:-1]
        for j in lhs:
            if j[0]!='_':
                assert onlyonesynset
                onlyonesynset = False
                for ll in synset2neg[j]:
                    for k in lhs:
                        if k[0]=='_':
                            posln[lemme2idx[k],currentidx]+=1/float(len(lhs))
                            posl[lemme2idx[k],currentidx]+=1/float(len(lhs))
                        else:
                            posln[synset2idx[ll],currentidx]+=1/float(len(lhs))
                            posl[synset2idx[k],currentidx]+=1/float(len(lhs))
                    for k in rhs:
                        posrn[lemme2idx[k],currentidx]+=1/float(len(rhs))
                        posr[lemme2idx[k],currentidx]+=1/float(len(rhs))
                    for k in rel:
                        poson[lemme2idx[k],currentidx]+=1/float(len(rel))
                        poso[lemme2idx[k],currentidx]+=1/float(len(rel))
                    currentidx+=1
        for j in rhs:
            if j[0]!='_':
                assert onlyonesynset
                onlyonesynset = False
                for ll in synset2neg[j]:
                    for k in rhs:
                        if k[0]=='_':
                            posrn[lemme2idx[k],currentidx]+=1/float(len(rhs))
                            posr[lemme2idx[k],currentidx]+=1/float(len(rhs))
                        else:
                            posrn[synset2idx[ll],currentidx]+=1/float(len(rhs))
                            posr[synset2idx[k],currentidx]+=1/float(len(rhs))
                    for k in lhs:
                        posln[lemme2idx[k],currentidx]+=1/float(len(lhs))
                        posl[lemme2idx[k],currentidx]+=1/float(len(lhs))
                    for k in rel:
                        poson[lemme2idx[k],currentidx]+=1/float(len(rel))
                        poso[lemme2idx[k],currentidx]+=1/float(len(rel))
                    currentidx+=1
        for j in rel:
            if j[0]!='_':
                assert onlyonesynset
                onlyonesynset = False
                for ll in synset2neg[j]:
                    for k in rel:
                        if k[0]=='_':
                            poson[lemme2idx[k],currentidx]+=1/float(len(rel))
                            poso[lemme2idx[k],currentidx]+=1/float(len(rel))
                        else:
                            poson[synset2idx[ll],currentidx]+=1/float(len(rel))
                            poso[synset2idx[k],currentidx]+=1/float(len(rel))
                    for k in rhs:
                        posrn[lemme2idx[k],currentidx]+=1/float(len(rhs))
                        posr[lemme2idx[k],currentidx]+=1/float(len(rhs))
                    for k in lhs:
                        posln[lemme2idx[k],currentidx]+=1/float(len(lhs))
                        posl[lemme2idx[k],currentidx]+=1/float(len(lhs))
                    currentidx+=1


    assert currentidx == ct

    #neworder = numpy.random.permutation(ct)

    poso = (poso.tocsr())#[:,neworder]
    posl = (posl.tocsr())#[:,neworder]
    posr = (posr.tocsr())#[:,neworder]

    poson = (poson.tocsr())#[:,neworder]
    posln = (posln.tocsr())#[:,neworder]
    posrn = (posrn.tocsr())#[:,neworder]


    f = open('Wikisuper-lhs.pkl','w')
    g = open('Wikisuper-rhs.pkl','w')
    h = open('Wikisuper-rel.pkl','w')

    i = open('Wikisuper-lhsn.pkl','w')
    j = open('Wikisuper-rhsn.pkl','w')
    k = open('Wikisuper-reln.pkl','w')


    cPickle.dump(posl,f,-1)
    cPickle.dump(posr,g,-1)
    cPickle.dump(poso,h,-1)

    cPickle.dump(posln,i,-1)
    cPickle.dump(posrn,j,-1)
    cPickle.dump(poson,k,-1)


    f.close()
    g.close()
    h.close()

    i.close()
    j.close()
    k.close()




f = open('/data/lisa/data/NLU/semcor3.0/brown-synsets/Brown-filtered-triplets-unambiguous-lemmas.dat','r')
g = open('/data/lisa/data/NLU/semcor3.0/brown-synsets/Brown-filtered-triplets-unambiguous-synsets.dat','r')

dat1 = f.readlines()
f.close()
dat2 = g.readlines()
g.close()

missed = 0
ct = 0
for i,k in zip(dat1,dat2):
    lhs,rel,rhs = parseline(i[:-1])
    lhsr,relr,rhsr = parseline(k[:-1])
    for j in lhs:
        if len(lemme2synset[j])>1:
            ct += (len(lemme2synset[j]))
        else:
            missed += 1
    j = rel[0]
    if len(lemme2synset[j])>1:
        ct += len(lemme2synset[j])
    else:
        missed += 1
    for j in rhs:
        if len(lemme2synset[j])>1:
            ct += len(lemme2synset[j])
        else:
            missed += 1
print ct,missed

import scipy.sparse
posl = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,ct),dtype='float32')
posr = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,ct),dtype='float32')
poso = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,ct),dtype='float32')

idxcurrent = 0
idxex = 0
dictidx ={}
freqlist = []
label = []
for i,k in zip(dat1,dat2):
    lhs,rel,rhs = parseline(i[:-1])
    lhsr,relr,rhsr = parseline(k[:-1])

    for idxtmp,k in enumerate(lhs):
        if len(lemme2synset[k])>1:
            dictidx.update({idxex:(idxcurrent,idxcurrent+len(lemme2synset[k]),lhsr[idxtmp] in synset2neg.keys())})
            for l,ff in zip(lemme2synset[k],lemme2freq[k]):
                for j in list(lhs[:idxtmp])+list(lhs[(idxtmp+1):]):
                    posl[lemme2idx[j],idxcurrent]+=1/float(len(lhs))
                posl[synset2idx[l],idxcurrent]+=1/float(len(lhs))
                freqlist+=[ff]
                if l == lhsr[idxtmp]:
                    label += [1]
                else:
                    label += [0]
                j = rel[0]
                poso[lemme2idx[j],idxcurrent]+=1
                for j in rhs:
                    posr[lemme2idx[j],idxcurrent]+=1/float(len(rhs))
                idxcurrent+=1
            idxex+=1
    k = rel[0]
    if len(lemme2synset[k])>1:
        dictidx.update({idxex:(idxcurrent,idxcurrent+len(lemme2synset[k]),relr[0] in synset2neg.keys())})
        for l,ff in zip(lemme2synset[k],lemme2freq[k]):
            for j in lhs:
                posl[lemme2idx[j],idxcurrent]+=1/float(len(lhs))
            poso[synset2idx[l],idxcurrent]+=1
            freqlist+=[ff]
            if l == relr[0]:
                label += [1]
            else:
                label += [0]
            for j in rhs:
                posr[lemme2idx[j],idxcurrent]+=1/float(len(rhs))
            idxcurrent+=1
        idxex+=1

    for idxtmp,k in enumerate(rhs):
        if len(lemme2synset[k])>1:
            dictidx.update({idxex:(idxcurrent,idxcurrent+len(lemme2synset[k]),rhsr[idxtmp] in synset2neg.keys())})
            for l,ff in zip(lemme2synset[k],lemme2freq[k]):
                for j in list(rhs[:idxtmp])+list(rhs[(idxtmp+1):]):
                    posr[lemme2idx[j],idxcurrent]+=1/float(len(rhs))
                posr[synset2idx[l],idxcurrent]+=1/float(len(rhs))
                freqlist+=[ff]
                if l == rhsr[idxtmp]:
                    label += [1]
                else:
                    label += [0]
                j = rel[0]
                poso[lemme2idx[j],idxcurrent]+=1
                for j in lhs:
                    posl[lemme2idx[j],idxcurrent]+=1/float(len(lhs))
                idxcurrent+=1
            idxex+=1

print idxcurrent,idxex,len(freqlist),len(dictidx),len(label),sum(label)
f = open('Brown-WSD-lhs.pkl','w')
g = open('Brown-WSD-rhs.pkl','w')
h = open('Brown-WSD-rel.pkl','w')
i = open('Brown-WSD-dict.pkl','w')
j = open('Brown-WSD-lab.pkl','w')
k = open('Brown-WSD-freq.pkl','w')

cPickle.dump(posl,f,-1)
f.close()
cPickle.dump(posr,g,-1)
g.close()
cPickle.dump(poso,h,-1)
h.close()
cPickle.dump(dictidx,i,-1)
i.close()
cPickle.dump(label,j,-1)
j.close()
cPickle.dump(freqlist,k,-1)
k.close()


import scipy.sparse
poslS = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,len(dat1)),dtype='float32')
posrS = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,len(dat1)),dtype='float32')
posoS = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,len(dat1)),dtype='float32')
poslL = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,len(dat1)),dtype='float32')
posrL = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,len(dat1)),dtype='float32')
posoL = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,len(dat1)),dtype='float32')


idxcurrent = 0
for i,k in zip(dat1,dat2):
    lhs,rel,rhs = parseline(i[:-1])
    lhsr,relr,rhsr = parseline(k[:-1])
    for j,k in zip(lhs,lhsr):
        poslL[lemme2idx[j],idxcurrent]+=1/float(len(lhs))
        poslS[synset2idx[k],idxcurrent]+=1/float(len(lhs))
    j = rel[0]
    k = relr[0]
    posoL[lemme2idx[j],idxcurrent]+=1
    posoS[synset2idx[k],idxcurrent]+=1
    for j in rhs:
        posrL[lemme2idx[j],idxcurrent]+=1/float(len(rhs))
        posrS[synset2idx[k],idxcurrent]+=1/float(len(rhs))
    idxcurrent+=1

f = open('Brown-lemme-lhs.pkl','w')
g = open('Brown-lemme-rhs.pkl','w')
h = open('Brown-lemme-rel.pkl','w')
i = open('Brown-synset-lhs.pkl','w')
j = open('Brown-synset-rhs.pkl','w')
k = open('Brown-synset-rel.pkl','w')

cPickle.dump(poslL,f,-1)
f.close()
cPickle.dump(posrL,g,-1)
g.close()
cPickle.dump(posoL,h,-1)
h.close()
cPickle.dump(poslS,i,-1)
i.close()
cPickle.dump(posrS,j,-1)
j.close()
cPickle.dump(posoS,k,-1)
k.close()
