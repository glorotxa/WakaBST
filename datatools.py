import numpy
import cPickle


### Create lemmes, synsets, indexes, definitions... dictionnaries ###
#########################################################################################################

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

for i in ['_PropertyOf','_MadeOf','_DefinedAs','_PartOf','_IsA','_UsedFor','_CapableOfReceivingAction','_LocationOf','_SubeventOf','_LastSubeventOf','_PrerequisiteEventOf','_FirstSubeventOf','_EffectOf','_DesirousEffectOf','_DesireOf','_MotivationOf','_CapableOf']:
    lemme2synset.update({i:[i]})
    lemme2freq.update({i:[1.0]})

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

#########################################################################################################




def parseline(line):
    lhs,rel,rhs = line.split('\t')
    lhs = lhs.split(' ')
    rhs = rhs.split(' ')
    rel = rel.split(' ')
    return lhs,rel,rhs






### Create WordNet3.0 sparse matrices of the lhs, rel and rhs ###
# 3 cases -> 
#     --synsets only (nolemme == 1),
#     --combinaisons consisting of 1 lemmes and others as synsets (nolemme == 2)
#     --lemmes only (nolemme == 3)
#########################################################################################################


if True:
    numpy.random.seed(753)
    # ignore the followoing WordNet relations (too few of them)
    speciallist = ['_substance_holonym','_attribute','_substance_meronym','_entailment','_cause']
    for nolemme in [1,2,3]:
        if nolemme==1:
            ll = ['train','val','test']
        else:
            ll = ['train']
        for datatyp in ll:
            f = open('/data/lisa/data/NLU/wordnet3.0-synsets/filtered-data/%s-WordNet3.0-filtered-synsets-relations-anto.txt'%datatyp,'r')

            dat = f.readlines()
            f.close()

            ct = 0
            for i in dat:
                lhs,rel,rhs = parseline(i[:-1])
                if rel[0] not in speciallist:
                    if nolemme==1 or nolemme==3:
                       ct += 1
                    if nolemme==2:
                        for j in synset2lemme[lhs[0]]:
                            if len(lemme2synset[j])!=1:
                                ct += 1
                        for j in synset2lemme[rel[0]]:
                            if len(lemme2synset[j])!=1:
                                assert False
                                ct += 1
                        for j in synset2lemme[rhs[0]]:
                            if len(lemme2synset[j])!=1:
                                ct += 1

            print len(dat),ct

            import scipy.sparse
            if datatyp == 'train':
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
                    if nolemme==1:
                        posl[synset2idx[lhs[0]],ct]=1
                        posr[synset2idx[rhs[0]],ct]=1
                        poso[synset2idx[rel[0]],ct]=1
                        ct+=1
                    if nolemme==3:
                        posl[lemme2idx[synset2lemme[lhs[0]][numpy.random.permutation(len(synset2lemme[lhs[0]]))[0]]],ct]=1
                        posr[lemme2idx[synset2lemme[rhs[0]][numpy.random.permutation(len(synset2lemme[rhs[0]]))[0]]],ct]=1
                        poso[lemme2idx[synset2lemme[rel[0]][numpy.random.permutation(len(synset2lemme[rel[0]]))[0]]],ct]=1
                        ct+=1
                    if nolemme==2:
                        for j in synset2lemme[lhs[0]]:
                            if len(lemme2synset[j])!=1: 
                                posl[lemme2idx[j],ct]=1
                                posr[synset2idx[rhs[0]],ct]=1
                                poso[synset2idx[rel[0]],ct]=1
                                ct += 1
                        for j in synset2lemme[rel[0]]:
                            if len(lemme2synset[j])!=1:
                                assert False
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
            
            if nolemme==1:
                f = open('WordNet3.0-%s-lhs.pkl'%datatyp,'w')
                g = open('WordNet3.0-%s-rhs.pkl'%datatyp,'w')
                h = open('WordNet3.0-%s-rel.pkl'%datatyp,'w')
            if nolemme==2:
                f = open('WordNet3.0-syle-%s-lhs.pkl'%datatyp,'w')
                g = open('WordNet3.0-syle-%s-rhs.pkl'%datatyp,'w')
                h = open('WordNet3.0-syle-%s-rel.pkl'%datatyp,'w')
            if nolemme==3:
                f = open('WordNet3.0-lemme-%s-lhs.pkl'%datatyp,'w')
                g = open('WordNet3.0-lemme-%s-rhs.pkl'%datatyp,'w')
                h = open('WordNet3.0-lemme-%s-rel.pkl'%datatyp,'w')
                
            cPickle.dump(posl.tocsr(),f,-1)
            cPickle.dump(posr.tocsr(),g,-1)
            cPickle.dump(poso.tocsr(),h,-1)

            f.close()
            g.close()
            h.close()

#########################################################################################################




### Create sampled Wikipedia sparse matrices of the lhs, rel and rhs ###
# over the 130 files + the unambiguous wiki.
# For each triplets of the 130 files: 
#       - we create one instance with lemmas
#       - we create one instance for each ambiguous lemmas that we replace by a synset sampled according to the frequencies
# For the unambiguous wiki: only synsets.
#########################################################################################################

if True:
    totalsize = 0
    for nbf in range(131):
        f = open('/data/lisa/data/NLU/converted-wikipedia/nlu-data/triplets-file-%s.dat'%nbf,'r')
        dat = f.readlines()
        f.close()
        for i in dat:
            totalsize +=1
            bb = False
            lhs,rel,rhs = parseline(i[:-1])
            if lhs[-1]=='':
                lhs = lhs[:-1]
            if rhs[-1]=='':
                rhs = rhs[:-1]
            if rel[-1]=='':
                rel = rel[:-1]
            for j in lhs:
                if len(lemme2synset[j])!=1:
                    bb = True
                    totalsize+=1
            for j in rhs:
                if len(lemme2synset[j])!=1:
                    bb = True
                    totalsize+=1
            for j in rel:
                if len(lemme2synset[j])!=1:
                    bb = True
                    totalsize+=1
            #assert bb
    f = open('/data/lisa/data/NLU/converted-wikipedia/nlu-data-synsets/unambiguous-triplets.dat','r')
    dat = f.readlines()
    totalsize+=len(dat)

    import scipy.sparse
    posl = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,totalsize),dtype='float32')
    posr = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,totalsize),dtype='float32')
    poso = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,totalsize),dtype='float32')
    
    numpy.random.seed(888)
    ct = 0
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
            for i in lhs:
                posl[lemme2idx[i],ct]+=1/float(len(lhs))
            for i in rel:
                poso[lemme2idx[i],ct]+=1/float(len(rel))
            for i in rhs:
                posr[lemme2idx[i],ct]+=1/float(len(rhs))
            ct+=1
            for idxtmp,k in enumerate(lhs):
                if len(lemme2synset[k])>1:
                    listfreqtmp = numpy.cumsum(lemme2freq[k])
                    idxcc = (list(listfreqtmp >= numpy.random.uniform())).index(True)
                    l = lemme2synset[k][idxcc]
                    for j in list(lhs[:idxtmp])+list(lhs[(idxtmp+1):]):
                        posl[lemme2idx[j],ct]+=1/float(len(lhs))
                    posl[synset2idx[l],ct]+=1/float(len(lhs))
                    for j in rel:
                        poso[lemme2idx[j],ct]+=1/float(len(rel))
                    for j in rhs:
                        posr[lemme2idx[j],ct]+=1/float(len(rhs))
                    ct+=1 
            for idxtmp,k in enumerate(rel):
                if len(lemme2synset[k])>1:
                    listfreqtmp = numpy.cumsum(lemme2freq[k])
                    idxcc = (list(listfreqtmp >= numpy.random.uniform())).index(True)
                    l = lemme2synset[k][idxcc]
                    for j in list(rel[:idxtmp])+list(rel[(idxtmp+1):]):
                        poso[lemme2idx[j],ct]+=1/float(len(rel))
                    poso[synset2idx[l],ct]+=1/float(len(rel))
                    for j in lhs:
                        posl[lemme2idx[j],ct]+=1/float(len(lhs))
                    for j in rhs:
                        posr[lemme2idx[j],ct]+=1/float(len(rhs))
                    ct+=1
            for idxtmp,k in enumerate(rhs):
                if len(lemme2synset[k])>1:
                    listfreqtmp = numpy.cumsum(lemme2freq[k])
                    idxcc = (list(listfreqtmp >= numpy.random.uniform())).index(True)
                    l = lemme2synset[k][idxcc]
                    for j in list(rhs[:idxtmp])+list(rhs[(idxtmp+1):]):
                        posr[lemme2idx[j],ct]+=1/float(len(rhs))
                    posr[synset2idx[l],ct]+=1/float(len(rhs))
                    for j in rel:
                        poso[lemme2idx[j],ct]+=1/float(len(rel))
                    for j in lhs:
                        posl[lemme2idx[j],ct]+=1/float(len(lhs))
                    ct+=1 

    f = open('/data/lisa/data/NLU/converted-wikipedia/nlu-data-synsets/unambiguous-triplets.dat','r')
    dat = f.readlines()
    for i in dat:
        lhs,rel,rhs = parseline(i[:-1])
        if lhs[-1]=='':
            lhs = lhs[:-1]
        if rhs[-1]=='':
            rhs = rhs[:-1]
        if rel[-1]=='':
            rel = rel[:-1]
        for j in lhs:
            posl[synset2idx[j],ct]+=1/float(len(lhs))
        for j in rhs:
            posr[synset2idx[j],ct]+=1/float(len(rhs))
        for j in rel:
            poso[synset2idx[j],ct]+=1/float(len(rel))
        ct += 1

    assert ct == totalsize
    print "finished"
    numpy.random.seed(999)
    neworder = numpy.random.permutation(totalsize)

    poso = (poso.tocsr())[:,neworder]
    posl = (posl.tocsr())[:,neworder]
    posr = (posr.tocsr())[:,neworder]

    f = open('Wikisample-lhs.pkl','w')
    g = open('Wikisample-rhs.pkl','w')
    h = open('Wikisample-rel.pkl','w')
    
    cPickle.dump(posl,f,-1)
    cPickle.dump(posr,g,-1)
    cPickle.dump(poso,h,-1)

    f.close()
    g.close()
    h.close()


#########################################################################################################


### Create Wikipedia sparse matrices of the lhs, rel and rhs ###
# choosing one synset randomly (following the frequences) for all ambiguous lemmas 
# over the 130 files + the unambiguous wiki
#########################################################################################################


if True:
    totalsize = 0
    for nbf in range(131):
        f = open('/data/lisa/data/NLU/converted-wikipedia/nlu-data/triplets-file-%s.dat'%nbf,'r')
        dat = f.readlines()
        f.close()
        totalsize += len(dat)
    f = open('/data/lisa/data/NLU/converted-wikipedia/nlu-data-synsets/unambiguous-triplets.dat','r')
    dat = f.readlines()
    totalsize+=len(dat)

    import scipy.sparse
    posl = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,totalsize),dtype='float32')
    posr = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,totalsize),dtype='float32')
    poso = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,totalsize),dtype='float32')
    
    numpy.random.seed(888)
    ct = 0
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
            for i in lhs:
                if len(lemme2synset[i])>1:
                    listfreqtmp = numpy.cumsum(lemme2freq[i])
                    idxcc = (list(listfreqtmp >= numpy.random.uniform())).index(True)
                    posl[synset2idx[lemme2synset[i][idxcc]],ct]+=1/float(len(lhs))
                else:
                    posl[lemme2idx[i],ct]+=1/float(len(lhs))
            for i in rel:
                if len(lemme2synset[i])>1:
                    listfreqtmp = numpy.cumsum(lemme2freq[i])
                    idxcc = (list(listfreqtmp >= numpy.random.uniform())).index(True)
                    poso[synset2idx[lemme2synset[i][idxcc]],ct]+=1/float(len(rel))
                else:
                    poso[lemme2idx[i],ct]+=1/float(len(rel))
            for i in rhs:
                if len(lemme2synset[i])>1:
                    listfreqtmp = numpy.cumsum(lemme2freq[i])
                    idxcc = (list(listfreqtmp >= numpy.random.uniform())).index(True)
                    posr[synset2idx[lemme2synset[i][idxcc]],ct]+=1/float(len(rhs))
                else:
                    posr[lemme2idx[i],ct]+=1/float(len(rhs))
            ct+=1

    f = open('/data/lisa/data/NLU/converted-wikipedia/nlu-data-synsets/unambiguous-triplets.dat','r')
    dat = f.readlines()
    for i in dat:
        lhs,rel,rhs = parseline(i[:-1])
        if lhs[-1]=='':
            lhs = lhs[:-1]
        if rhs[-1]=='':
            rhs = rhs[:-1]
        if rel[-1]=='':
            rel = rel[:-1]
        for j in lhs:
            posl[synset2idx[j],ct]+=1/float(len(lhs))
        for j in rhs:
            posr[synset2idx[j],ct]+=1/float(len(rhs))
        for j in rel:
            poso[synset2idx[j],ct]+=1/float(len(rel))
        ct += 1

    assert ct == totalsize
    print "finished"
    numpy.random.seed(999)
    neworder = numpy.random.permutation(totalsize)

    poso = (poso.tocsr())[:,neworder]
    posl = (posl.tocsr())[:,neworder]
    posr = (posr.tocsr())[:,neworder]

    f = open('Wikisamplesy-lhs.pkl','w')
    g = open('Wikisamplesy-rhs.pkl','w')
    h = open('Wikisamplesy-rel.pkl','w')
    
    cPickle.dump(posl,f,-1)
    cPickle.dump(posr,g,-1)
    cPickle.dump(poso,h,-1)

    f.close()
    g.close()
    h.close()


#########################################################################################################


### Create Supervised Wikipedia (using N. Usunier trick) sparse matrices of the lhs, rel and rhs ###
#########################################################################################################

if True:
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
    
    numpy.random.seed(222) 
    neworder = numpy.random.permutation(len(dat))
    currentidx = 0
    numpy.random.seed(666)
    for iii in neworder:
        i = dat[iii]
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

#########################################################################################################


### Create the WSD test set for the Brown corpus ###
### lhs,rel,rhs -> the data with all the possible synset configurations for unambiguous lemmas
### dict -> dictionnary of the form {index_beginning:index_end,...} (for each ambiguous lemma to disambiguate).
### lab -> vector of length = number of instances to score (1 correspond to the real synset, 0 elsewhere)
### freq -> vector of frequencies of the synsets in consideration (with respect to the given lemme)
#########################################################################################################

if True:
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

    ### Also create the normal Brown corpus lhs,rel,rhs sparse matrices ###
    ### lemme: only lemmas
    ### synset: only synset
    ### corres: fill the sparse matrices in the following way -> mat[lemmeidx,instanceidx]=synsetidx
    ###         to keep track of the correspondences
    #########################################################################################################

    import scipy.sparse
    poslS = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,len(dat1)),dtype='float32')
    posrS = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,len(dat1)),dtype='float32')
    posoS = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,len(dat1)),dtype='float32')
    poslL = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,len(dat1)),dtype='float32')
    posrL = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,len(dat1)),dtype='float32')
    posoL = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,len(dat1)),dtype='float32')
    poslLS = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,len(dat1)),dtype='float32')
    posrLS = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,len(dat1)),dtype='float32')
    posoLS = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,len(dat1)),dtype='float32')

    idxcurrent = 0
    for i,k in zip(dat1,dat2):
        lhs,rel,rhs = parseline(i[:-1])
        lhsr,relr,rhsr = parseline(k[:-1])
        for j,k in zip(lhs,lhsr):
            poslL[lemme2idx[j],idxcurrent]+=1/float(len(lhs))
            poslS[synset2idx[k],idxcurrent]+=1/float(len(lhs))
            poslLS[lemme2idx[j],idxcurrent] = synset2idx[k]
        j = rel[0]
        k = relr[0]
        posoL[lemme2idx[j],idxcurrent]+=1
        posoS[synset2idx[k],idxcurrent]+=1
        posoLS[lemme2idx[j],idxcurrent] = synset2idx[k]
        for j,k in zip(rhs,rhsr):
            posrL[lemme2idx[j],idxcurrent]+=1/float(len(rhs))
            posrS[synset2idx[k],idxcurrent]+=1/float(len(rhs))
            posrLS[lemme2idx[j],idxcurrent] = synset2idx[k]
        idxcurrent+=1

    f = open('Brown-lemme-lhs.pkl','w')
    g = open('Brown-lemme-rhs.pkl','w')
    h = open('Brown-lemme-rel.pkl','w')
    i = open('Brown-synset-lhs.pkl','w')
    j = open('Brown-synset-rhs.pkl','w')
    k = open('Brown-synset-rel.pkl','w')
    l = open('Brown-corres-lhs.pkl','w')
    m = open('Brown-corres-rhs.pkl','w')
    n = open('Brown-corres-rel.pkl','w')

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
    cPickle.dump(poslLS,l,-1)
    l.close()
    cPickle.dump(posrLS,m,-1)
    m.close()
    cPickle.dump(posoLS,n,-1)
    n.close()

#########################################################################################################



### Create the XWN WSD test set with concept name (asked by antoine) ###
#########################################################################################################

if False:
    g = open('/data/lisa/data/NLU/XWN/extended-wordnet-filtered-synsets.txt','r')
    dd = open('/data/lisa/data/NLU/XWN/extended-wordnet-test.txt','w')

    dat = g.readlines()
    g.close()
    numpy.random.seed(468)
    order = numpy.random.permutation(len(dat))
    txt = ''
    missed = 0
    ct = 0
    for ii in range(5000):
        k = dat[order[ii]]
        lhsr,relr,rhsr = parseline(k[:-1])
        bf = True
        for j in lhsr:
            if not bf:
                txt +=' '
            else:
                bf = False
            txt+= str(synset2concept[j])
        txt+='\t'
        bf = True
        for j in relr:
            if not bf:
                txt +=' '
            else:
                bf = False
            txt+= str(synset2concept[j])
        txt+='\t'
        bf = True
        for j in rhsr:
            if not bf:
                txt +=' '
            else:
                bf = False
            txt+= str(synset2concept[j])
        txt+='\n'
    dd.write(txt)
    dd.close()

#########################################################################################################


### Create the WSD test set for the XWN corpus ###
### lhs,rel,rhs -> the data with all the possible synset configurations for unambiguous lemmas
### dict -> dictionnary of the form {index_beginning:index_end,...} (for each ambiguous lemma to disambiguate).
### lab -> vector of length = number of instances to score (1 correspond to the real synset, 0 elsewhere)
### freq -> vector of frequencies of the synsets in consideration (with respect to the given lemme)
### uncomment to create data corresponding to the model choice (mod) and a different synset than the model choice and the true label (nmod)
### need to have done an evaluation of a trained model (see evaluation.py)
#########################################################################################################

if True:
    f = open('/data/lisa/data/NLU/XWN/extended-wordnet-filtered-lemmas.txt','r')
    g = open('/data/lisa/data/NLU/XWN/extended-wordnet-filtered-synsets.txt','r')

    dat1 = f.readlines()
    f.close()
    dat2 = g.readlines()
    g.close()
    numpy.random.seed(468)
    order = numpy.random.permutation(len(dat1))
    
    missed = 0
    ct = 0
    for ii in range(5000):
        i = dat1[order[ii]]
        k = dat2[order[ii]]
        lhs,rel,rhs = parseline(i[:-1])
        lhsr,relr,rhsr = parseline(k[:-1])
        for j in lhs:
            if len(lemme2synset[j])>1:
                ct += (len(lemme2synset[j]))
            else:
                missed += 1
        for j in rel:
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
    #poslm = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,5000),dtype='float32')
    #posrm = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,5000),dtype='float32')
    #posom = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,5000),dtype='float32')
    #poslnm = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,5000),dtype='float32')
    #posrnm = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,5000),dtype='float32')
    #posonm = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,5000),dtype='float32')

    #llmodel = cPickle.load(open('modelpred.pkl')) 
    #nllmodel = cPickle.load(open('nmodelpred.pkl')) 
    idxcurrent = 0
    idxex = 0
    dictidx ={}
    freqlist = []
    label = []
    for ii in range(5000):
        i = dat1[order[ii]]
        k = dat2[order[ii]]
        lhs,rel,rhs = parseline(i[:-1])
        lhsr,relr,rhsr = parseline(k[:-1])

        for idxtmp,k in enumerate(lhs):
            if len(lemme2synset[k])>1:
                #poslm[lemme2idx[k],ii]=synset2idx[lemme2synset[k][llmodel[idxex]]]
                #poslnm[lemme2idx[k],ii]=synset2idx[lemme2synset[k][nllmodel[idxex]]]
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
                    for j in rel:
                        poso[lemme2idx[j],idxcurrent]+=1/float(len(rel))
                    for j in rhs:
                        posr[lemme2idx[j],idxcurrent]+=1/float(len(rhs))
                    idxcurrent+=1
                idxex+=1
            else:
                pass
                #poslm[lemme2idx[k],ii]=lemme2idx[k]
                #poslnm[lemme2idx[k],ii]=lemme2idx[k]
        
        for idxtmp,k in enumerate(rel):
            if len(lemme2synset[k])>1:
                #posom[lemme2idx[k],ii]=synset2idx[lemme2synset[k][llmodel[idxex]]]
                #posonm[lemme2idx[k],ii]=synset2idx[lemme2synset[k][nllmodel[idxex]]]
                dictidx.update({idxex:(idxcurrent,idxcurrent+len(lemme2synset[k]),relr[idxtmp] in synset2neg.keys())})
                for l,ff in zip(lemme2synset[k],lemme2freq[k]):
                    for j in lhs:
                        posl[lemme2idx[j],idxcurrent]+=1/float(len(lhs))
                    for j in list(rel[:idxtmp])+list(rel[(idxtmp+1):]):
                        poso[lemme2idx[j],idxcurrent]+=1/float(len(rel))
                    poso[synset2idx[l],idxcurrent]+=1/float(len(rel))
                    freqlist+=[ff]
                    if l == relr[idxtmp]:
                        label += [1]
                    else:
                        label += [0]
                    for j in rhs:
                        posr[lemme2idx[j],idxcurrent]+=1/float(len(rhs))
                    idxcurrent+=1
                idxex+=1
            else:
                pass
                #posom[lemme2idx[k],ii]=lemme2idx[k]
                #posonm[lemme2idx[k],ii]=lemme2idx[k]

        for idxtmp,k in enumerate(rhs):
            if len(lemme2synset[k])>1:
                #posrm[lemme2idx[k],ii]=synset2idx[lemme2synset[k][llmodel[idxex]]]
                #posrnm[lemme2idx[k],ii]=synset2idx[lemme2synset[k][nllmodel[idxex]]]
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
                    for j in rel:
                        poso[lemme2idx[j],idxcurrent]+=1/float(len(rel))
                    for j in lhs:
                        posl[lemme2idx[j],idxcurrent]+=1/float(len(lhs))
                    idxcurrent+=1
                idxex+=1
            else:
                pass
                #posrm[lemme2idx[k],ii]=lemme2idx[k]
                #posrnm[lemme2idx[k],ii]=lemme2idx[k]

    print idxcurrent,idxex,len(freqlist),len(dictidx),len(label),sum(label)
    f = open('XWN-WSD-lhs.pkl','w')
    g = open('XWN-WSD-rhs.pkl','w')
    h = open('XWN-WSD-rel.pkl','w')
    i = open('XWN-WSD-dict.pkl','w')
    j = open('XWN-WSD-lab.pkl','w')
    k = open('XWN-WSD-freq.pkl','w')
    #l = open('XWN-mod-lhs.pkl','w')
    #m = open('XWN-mod-rhs.pkl','w')
    #n = open('XWN-mod-rel.pkl','w')
    #o = open('XWN-nmod-lhs.pkl','w')
    #p = open('XWN-nmod-rhs.pkl','w')
    #q = open('XWN-nmod-rel.pkl','w')
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
    #cPickle.dump(poslm,l,-1)
    #l.close()
    #cPickle.dump(posrm,m,-1)
    #m.close()
    #cPickle.dump(posom,n,-1)
    #n.close()
    #cPickle.dump(poslnm,o,-1)
    #o.close()
    #cPickle.dump(posrnm,p,-1)
    #p.close()
    #cPickle.dump(posonm,q,-1)
    #q.close()


#########################################################################################################


### Create the normal XWN test corpus lhs,rel,rhs sparse matrices ###
### lemme: only lemmas
### synset: only synset
### corres: fill the sparse matrices in the following way -> mat[lemmeidx,instanceidx]=synsetidx
###         to keep track of the correspondences
#########################################################################################################


if True:
    import scipy.sparse
    poslS = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,5000),dtype='float32')
    posrS = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,5000),dtype='float32')
    posoS = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,5000),dtype='float32')
    poslL = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,5000),dtype='float32')
    posrL = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,5000),dtype='float32')
    posoL = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,5000),dtype='float32')
    poslLS = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,5000),dtype='float32')
    posrLS = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,5000),dtype='float32')
    posoLS = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,5000),dtype='float32')
    
    idxcurrent = 0
    for ii in range(5000):
        i = dat1[order[ii]]
        k = dat2[order[ii]]
        lhs,rel,rhs = parseline(i[:-1])
        lhsr,relr,rhsr = parseline(k[:-1])
        for j,k in zip(lhs,lhsr):
            poslL[lemme2idx[j],idxcurrent]+=1/float(len(lhs))
            poslS[synset2idx[k],idxcurrent]+=1/float(len(lhs))
            poslLS[lemme2idx[j],idxcurrent] = synset2idx[k]
        for j,k in zip(rel,relr):
            posoL[lemme2idx[j],idxcurrent]+=1/float(len(rel))
            posoS[synset2idx[k],idxcurrent]+=1/float(len(rel))
            posoLS[lemme2idx[j],idxcurrent] = synset2idx[k]
        for j,k in zip(rhs,rhsr):
            posrL[lemme2idx[j],idxcurrent]+=1/float(len(rhs))
            posrS[synset2idx[k],idxcurrent]+=1/float(len(rhs))
            posrLS[lemme2idx[j],idxcurrent] = synset2idx[k]
        idxcurrent+=1

    f = open('XWN-lemme-lhs.pkl','w')
    g = open('XWN-lemme-rhs.pkl','w')
    h = open('XWN-lemme-rel.pkl','w')
    i = open('XWN-synset-lhs.pkl','w')
    j = open('XWN-synset-rhs.pkl','w')
    k = open('XWN-synset-rel.pkl','w')
    l = open('XWN-corres-lhs.pkl','w')
    m = open('XWN-corres-rhs.pkl','w')
    n = open('XWN-corres-rel.pkl','w')

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
    cPickle.dump(poslLS,l,-1)
    l.close()
    cPickle.dump(posrLS,m,-1)
    m.close()
    cPickle.dump(posoLS,n,-1)
    n.close()

    ### Also create the training XWN corpus lhs,rel,rhs sparse matrices ###
    # Supervised: instances of the type: 1 synset, others as lemmas
    #########################################################################################################

    ct = 0
    for ii in xrange(5000,len(dat1)):
        i = dat1[order[ii]]
        lhs,rel,rhs = parseline(i[:-1])
        for j in lhs:
            if len(lemme2synset[j])>1:
                ct += len(lemme2synset[j]) - 1 
        for j in rhs:
            if len(lemme2synset[j])>1:
                ct += len(lemme2synset[j]) - 1
        for j in rel:
            if len(lemme2synset[j])>1:
                ct += len(lemme2synset[j]) - 1
    import scipy.sparse

    posln = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,ct),dtype='float32')
    posrn = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,ct),dtype='float32')
    poson = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,ct),dtype='float32')
    posl = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,ct),dtype='float32')
    posr = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,ct),dtype='float32')
    poso = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,ct),dtype='float32')
    print len(dat1),ct

    ct = 0
    import copy
    for ii in xrange(5000,len(dat1)):
        i = dat1[order[ii]]
        k = dat2[order[ii]]
        lhs,rel,rhs = parseline(i[:-1])
        lhsr,relr,rhsr = parseline(k[:-1])
        for j,l in zip(lhs,lhsr):
            if len(lemme2synset[j])>1:
                lltmp1 = copy.deepcopy(lhs)
                lltmp1.remove(j)
                lltmp2 = copy.deepcopy(lemme2synset[j])
                lltmp2.remove(l)
                for bb in lltmp2:
                    for vv in lltmp1:
                        posl[lemme2idx[vv],ct]+=1/float(len(lhs))
                        posln[lemme2idx[vv],ct]+=1/float(len(lhs))
                    posl[synset2idx[l],ct]+=1/float(len(lhs))
                    posln[synset2idx[bb],ct]+=1/float(len(lhs))
                    for vv in rel:
                        poso[lemme2idx[vv],ct]+=1/float(len(rel))
                        poson[lemme2idx[vv],ct]+=1/float(len(rel))
                    for vv in rhs:
                        posr[lemme2idx[vv],ct]+=1/float(len(rhs))
                        posrn[lemme2idx[vv],ct]+=1/float(len(rhs))
                    ct += 1
        for j,l in zip(rhs,rhsr):
            if len(lemme2synset[j])>1:
                lltmp1 = copy.deepcopy(rhs)
                lltmp1.remove(j)
                lltmp2 = copy.deepcopy(lemme2synset[j])
                lltmp2.remove(l)
                for bb in lltmp2:
                    for vv in lltmp1:
                        posr[lemme2idx[vv],ct]+=1/float(len(rhs))
                        posrn[lemme2idx[vv],ct]+=1/float(len(rhs))
                    posr[synset2idx[l],ct]+=1/float(len(rhs))
                    posrn[synset2idx[bb],ct]+=1/float(len(rhs))
                    for vv in rel:
                        poso[lemme2idx[vv],ct]+=1/float(len(rel))
                        poson[lemme2idx[vv],ct]+=1/float(len(rel))
                    for vv in lhs:
                        posl[lemme2idx[vv],ct]+=1/float(len(lhs))
                        posln[lemme2idx[vv],ct]+=1/float(len(lhs))
                    ct += 1
        for j,l in zip(rel,relr):
            if len(lemme2synset[j])>1:
                lltmp1 = copy.deepcopy(rel)
                lltmp1.remove(j)
                lltmp2 = copy.deepcopy(lemme2synset[j])
                lltmp2.remove(l)
                for bb in lltmp2:
                    for vv in lltmp1:
                        poso[lemme2idx[vv],ct]+=1/float(len(rel))
                        poson[lemme2idx[vv],ct]+=1/float(len(rel))
                    poso[synset2idx[l],ct]+=1/float(len(rel))
                    poson[synset2idx[bb],ct]+=1/float(len(rel))
                    for vv in lhs:
                        posl[lemme2idx[vv],ct]+=1/float(len(lhs))
                        posln[lemme2idx[vv],ct]+=1/float(len(lhs))
                    for vv in rhs:
                        posr[lemme2idx[vv],ct]+=1/float(len(rhs))
                        posrn[lemme2idx[vv],ct]+=1/float(len(rhs))
                    ct += 1
    f = open('XWN-lhs.pkl','w')
    g = open('XWN-rhs.pkl','w')
    h = open('XWN-rel.pkl','w')
    i = open('XWN-lhsn.pkl','w')
    j = open('XWN-rhsn.pkl','w')
    k = open('XWN-reln.pkl','w')

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


#########################################################################################################


### Create the ConceptNet corpus ###
# Preprocessing with 3 filters: 
# 1 -> all elements have an unambiguous POS.
# 2 -> at least one element with unambigous POS for each members (ignore ambiguous POS).
# 3 -> at least one element for each member and report ambiguous POS by taking the most frequent one. 
#########################################################################################################


if True:
    from nltk.stem.wordnet import WordNetLemmatizer
    lmtzr = WordNetLemmatizer()
    f = open('/data/lisa/data/NLU/ConceptNet/predicates_concise_nonkline.txt','r')
    dat = f.readlines()
    g = open('/data/lisa/data/NLU/ConceptNet/predicates_concise_nonkline.txt','r')
    dat += g.readlines()
    f.close()
    g.close()
    ex = []
    for i in dat:
        print len(ex)
        rel,dum,couple = (i[1:-3]).partition(' ')
        rel = '_'+rel
        lcouple = couple[1:-1].split('" "')[:-1]
        lcouple[0] = lcouple[0].split(' ')
        left = []
        booltmp = True
        for j in range(len(couple[0])):
            lcouple[0][j] = lmtzr.lemmatize(lcouple[0][j])
            ctit = 0
            name = ''
            if '__' + lcouple[0][j] + '_NN' in lemme2idx.keys():
                ctit += 1 
                name ='__' +  lcouple[0][j] + '_NN'
            if '__' + lcouple[0][j] + '_VB' in lemme2idx.keys():
                ctit += 1
                name = '__' + lcouple[0][j] + '_VB'
            if '__' + lcouple[0][j] + '_JJ' in lemme2idx.keys():
                ctit += 1
                name = '__' + lcouple[0][j] + '_JJ'
            if '__' + lcouple[0][j] + '_RB' in lemme2idx.keys():
                ctit += 1
                name = '__' + lcouple[0][j] + '_RB'
            if ctit == 1:
                left += [name]
            else:
                booltmp = False
        #print 'left',lcouple[0],ctit,left
        lcouple[1] = lcouple[1].split(' ')
        right =[]
        for j in range(len(couple[1])):
            lcouple[1][j] = lmtzr.lemmatize(lcouple[1][j])
            ctit = 0
            name = ''
            if '__' + lcouple[1][j] + '_NN' in lemme2idx.keys():
                ctit += 1
                name ='__' +  lcouple[1][j] + '_NN'
            if '__' + lcouple[1][j] + '_VB' in lemme2idx.keys():
                ctit += 1
                name = '__' + lcouple[1][j] + '_VB'
            if '__' + lcouple[1][j] + '_JJ' in lemme2idx.keys():
                ctit += 1
                name = '__' + lcouple[1][j] + '_JJ'
            if '__' + lcouple[1][j] + '_RB' in lemme2idx.keys():
                ctit += 1
                name = '__' + lcouple[1][j] + '_RB'
            if ctit == 1:
                right += [name]
            else:
                booltmp = False
        #print 'right',lcouple[1],ctit,right
        if len(left)>=1 and len(right)>=1 and booltmp:
            ex += [[left,[rel],right]]
    print ex
    print numpy.max(lemme2idx.values())+1,len(ex)
    import scipy.sparse
    posl = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,len(ex)),dtype='float32')
    posr = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,len(ex)),dtype='float32')
    poso = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,len(ex)),dtype='float32')
    for ct,i in enumerate(ex):
        for j in i[0]:
            posl[lemme2idx[j],ct] += 1/float(len(i[0]))
        for j in i[1]:
            poso[lemme2idx[j],ct] += 1/float(len(i[1]))
        for j in i[2]:
            posr[lemme2idx[j],ct] += 1/float(len(i[2]))
    
    f = open('ConceptNet-lhs.pkl','w')
    g = open('ConceptNet-rhs.pkl','w')
    h = open('ConceptNet-rel.pkl','w')

    cPickle.dump(posl,f,-1)
    f.close()
    cPickle.dump(posr,g,-1)
    g.close()
    cPickle.dump(poso,h,-1)
    h.close()




if True:
    from nltk.stem.wordnet import WordNetLemmatizer
    lmtzr = WordNetLemmatizer()
    f = open('/data/lisa/data/NLU/ConceptNet/predicates_concise_nonkline.txt','r')
    dat = f.readlines()
    g = open('/data/lisa/data/NLU/ConceptNet/predicates_concise_nonkline.txt','r')
    dat += g.readlines()
    f.close()
    g.close()
    ex = []
    for i in dat:
        print len(ex)
        rel,dum,couple = (i[1:-3]).partition(' ')
        rel = '_'+rel
        lcouple = couple[1:-1].split('" "')[:-1]
        lcouple[0] = lcouple[0].split(' ')
        left = []
        booltmp = True
        for j in range(len(couple[0])):
            lcouple[0][j] = lmtzr.lemmatize(lcouple[0][j])
            ctit = 0
            name = ''
            if '__' + lcouple[0][j] + '_NN' in lemme2idx.keys():
                ctit += 1 
                name ='__' +  lcouple[0][j] + '_NN'
            if '__' + lcouple[0][j] + '_VB' in lemme2idx.keys():
                ctit += 1
                name = '__' + lcouple[0][j] + '_VB'
            if '__' + lcouple[0][j] + '_JJ' in lemme2idx.keys():
                ctit += 1
                name = '__' + lcouple[0][j] + '_JJ'
            if '__' + lcouple[0][j] + '_RB' in lemme2idx.keys():
                ctit += 1
                name = '__' + lcouple[0][j] + '_RB'
            if ctit == 1:
                left += [name]
            else:
                booltmp = False
        #print 'left',lcouple[0],ctit,left
        lcouple[1] = lcouple[1].split(' ')
        right =[]
        for j in range(len(couple[1])):
            lcouple[1][j] = lmtzr.lemmatize(lcouple[1][j])
            ctit = 0
            name = ''
            if '__' + lcouple[1][j] + '_NN' in lemme2idx.keys():
                ctit += 1
                name ='__' +  lcouple[1][j] + '_NN'
            if '__' + lcouple[1][j] + '_VB' in lemme2idx.keys():
                ctit += 1
                name = '__' + lcouple[1][j] + '_VB'
            if '__' + lcouple[1][j] + '_JJ' in lemme2idx.keys():
                ctit += 1
                name = '__' + lcouple[1][j] + '_JJ'
            if '__' + lcouple[1][j] + '_RB' in lemme2idx.keys():
                ctit += 1
                name = '__' + lcouple[1][j] + '_RB'
            if ctit == 1:
                right += [name]
            else:
                booltmp = False
        #print 'right',lcouple[1],ctit,right
        if len(left)>=1 and len(right)>=1:
            ex += [[left,[rel],right]]
    print ex
    print numpy.max(lemme2idx.values())+1,len(ex)
    import scipy.sparse
    posl = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,len(ex)),dtype='float32')
    posr = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,len(ex)),dtype='float32')
    poso = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,len(ex)),dtype='float32')
    for ct,i in enumerate(ex):
        for j in i[0]:
            posl[lemme2idx[j],ct] += 1/float(len(i[0]))
        for j in i[1]:
            poso[lemme2idx[j],ct] += 1/float(len(i[1]))
        for j in i[2]:
            posr[lemme2idx[j],ct] += 1/float(len(i[2]))
    
    f = open('ConceptNet2-lhs.pkl','w')
    g = open('ConceptNet2-rhs.pkl','w')
    h = open('ConceptNet2-rel.pkl','w')

    cPickle.dump(posl,f,-1)
    f.close()
    cPickle.dump(posr,g,-1)
    g.close()
    cPickle.dump(poso,h,-1)
    h.close()




if True:
    from nltk.stem.wordnet import WordNetLemmatizer
    lmtzr = WordNetLemmatizer()
    f = open('/data/lisa/data/NLU/ConceptNet/predicates_concise_nonkline.txt','r')
    dat = f.readlines()
    g = open('/data/lisa/data/NLU/ConceptNet/predicates_concise_nonkline.txt','r')
    dat += g.readlines()
    f.close()
    g.close()
    ex = []
    for i in dat:
        print len(ex)
        rel,dum,couple = (i[1:-3]).partition(' ')
        rel = '_'+rel
        lcouple = couple[1:-1].split('" "')[:-1]
        lcouple[0] = lcouple[0].split(' ')
        left = []
        booltmp = True
        for j in range(len(couple[0])):
            lcouple[0][j] = lmtzr.lemmatize(lcouple[0][j])
            ctit = 0
            name = ''
            if '__' + lcouple[0][j] + '_RB' in lemme2idx.keys():
                ctit += 1 
                name ='__' +  lcouple[0][j] + '_RB'
            if '__' + lcouple[0][j] + '_JJ' in lemme2idx.keys():
                ctit += 1
                name = '__' + lcouple[0][j] + '_JJ'
            if '__' + lcouple[0][j] + '_VB' in lemme2idx.keys():
                ctit += 1
                name = '__' + lcouple[0][j] + '_VB'
            if '__' + lcouple[0][j] + '_NN' in lemme2idx.keys():
                ctit += 1
                name = '__' + lcouple[0][j] + '_NN'
            if ctit > 0:
                left += [name]
            else:
                booltmp = False
        #print 'left',lcouple[0],ctit,left
        lcouple[1] = lcouple[1].split(' ')
        right =[]
        for j in range(len(couple[1])):
            lcouple[1][j] = lmtzr.lemmatize(lcouple[1][j])
            ctit = 0
            name = ''
            if '__' + lcouple[1][j] + '_RB' in lemme2idx.keys():
                ctit += 1
                name ='__' +  lcouple[1][j] + '_RB'
            if '__' + lcouple[1][j] + '_JJ' in lemme2idx.keys():
                ctit += 1
                name = '__' + lcouple[1][j] + '_JJ'
            if '__' + lcouple[1][j] + '_VB' in lemme2idx.keys():
                ctit += 1
                name = '__' + lcouple[1][j] + '_VB'
            if '__' + lcouple[1][j] + '_NN' in lemme2idx.keys():
                ctit += 1
                name = '__' + lcouple[1][j] + '_NN'
            if ctit > 0:
                right += [name]
            else:
                booltmp = False
        #print 'right',lcouple[1],ctit,right
        if len(left)>=1 and len(right)>=1:
            ex += [[left,[rel],right]]
    print ex
    print numpy.max(lemme2idx.values())+1,len(ex)
    import scipy.sparse
    posl = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,len(ex)),dtype='float32')
    posr = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,len(ex)),dtype='float32')
    poso = scipy.sparse.lil_matrix((numpy.max(lemme2idx.values())+1,len(ex)),dtype='float32')
    for ct,i in enumerate(ex):
        for j in i[0]:
            posl[lemme2idx[j],ct] += 1/float(len(i[0]))
        for j in i[1]:
            poso[lemme2idx[j],ct] += 1/float(len(i[1]))
        for j in i[2]:
            posr[lemme2idx[j],ct] += 1/float(len(i[2]))
    
    f = open('ConceptNet3-lhs.pkl','w')
    g = open('ConceptNet3-rhs.pkl','w')
    h = open('ConceptNet3-rel.pkl','w')

    cPickle.dump(posl,f,-1)
    f.close()
    cPickle.dump(posr,g,-1)
    g.close()
    cPickle.dump(poso,h,-1)
    h.close()


#########################################################################################################


### Create the Senseval3.0 test set ###
### lhs,rel,rhs -> the data with all the possible synset configurations for unambiguous lemmas
### dict -> dictionnary of the form {index_beginning:index_end,...} (for each ambiguous lemma to disambiguate).
### lab -> vector of length = number of instances to score (1 correspond to the real synset, 0 elsewhere)
### freq -> vector of frequencies of the synsets in consideration (with respect to the given lemme)
#########################################################################################################




if True:
    f = open('/data/lisa/data/NLU/senseval3/Senseval3-wn3.0-filtered-triplets-unambiguous-lemmas.dat','r')
    g = open('/data/lisa/data/NLU/senseval3/Senseval3-wn3.0-filtered-triplets-unambiguous-synsets.dat','r')

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
    f = open('Senseval3-WSD-lhs.pkl','w')
    g = open('Senseval3-WSD-rhs.pkl','w')
    h = open('Senseval3-WSD-rel.pkl','w')
    i = open('Senseval3-WSD-dict.pkl','w')
    j = open('Senseval3-WSD-lab.pkl','w')
    k = open('Senseval3-WSD-freq.pkl','w')

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
