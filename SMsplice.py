import numpy as np
from sklearn.neighbors import KernelDensity
from awkde import GaussianKDE

def baseToInt(base):
    if base == 'a': return 0
    elif base == 'c': return 1
    elif base == 'g': return 2
    elif base == 't': return 3
    else:
        print("nonstandard base encountered:", base)
        return -1
        
def hashSequence(seq):
    sum = 0 
    l = len(seq)
    for i in range(l):
        sum += (4**(l-i-1))*baseToInt(seq[i])
    return sum

def intToBase(i):
    if i == 0: return 'a'
    elif i == 1: return 'c'
    elif i == 2: return 'g'
    elif i == 3: return 't'
    else: 
        print("nonbase integer encountered:", i)
        return ''
    
def unhashSequence(num, l):
    seq = ''
    for i in range(l):
        seq += intToBase(num // 4**(l-i-1))
        num -= (num // 4**(l-i-1))*(4**(l-i-1))
    return seq

def trueSequencesCannonical(genes, annotations, E = 0, I = 1, B3 = 3, B5 = 5):
    # Converts gene annotations to sequences of integers indicating whether the sequence is exonic, intronic, or splice site,
    # Inputs
    #   - genes: a biopython style dictionary of the gene sequences
    #   - annotations: the splicing annotations dictionary
    #   - E, I, B3, B5: the integer indicators for exon, intron, 3'ss, and 5'ss, respectively
    trueSeqs = {}
    for gene in annotations.keys():
        if gene not in genes.keys(): 
            print(gene, 'has annotation, but was not found in the fasta file of genes') 
            continue
        
        transcript = annotations[gene]
        if len(transcript) == 1: 
            trueSeqs[gene] = np.zeros(len(genes[gene]), dtype = int) + E
            continue # skip the rest for a single exon case
        
        # First exon 
        true = np.zeros(len(genes[gene]), dtype = int) + I
        three = transcript[0][0] - 1 # Marking the beginning of the first exon
        five = transcript[0][1] + 1
        true[range(three+1, five)] = E
        true[five] = B5
        
        # Internal exons 
        for exon in transcript[1:-1]:
            three = exon[0] - 1
            five = exon[1] + 1
            true[three] = B3
            true[five] = B5
            true[range(three+1, five)] = E
            
        # Last exon 
        three = transcript[-1][0] - 1
        true[three] = B3
        five = transcript[-1][1] + 1 # Marking the end of the last exon
        true[range(three+1, five)] = E
                
        trueSeqs[gene] = true
        
    return(trueSeqs)

def trainAllTriplets(sequences, cutoff = 10**(-5)):
    # Train maximum entropy models from input sequences with triplet conditions
    train = np.zeros((len(sequences),len(sequences[0])), dtype = int)
    for (i, seq) in enumerate(sequences):
        for j in range(len(seq)):
            train[i,j] = baseToInt(seq[j])
    prob = np.log(np.zeros(4**len(sequences[0])) + 4**(-len(sequences[0])))
    Hprev = -np.sum(prob*np.exp(prob))/np.log(2)
    H = -1
    sequences = np.zeros((4**len(sequences[0]),len(sequences[0])), dtype = int)
    l = len(sequences[0]) - 1 
    for i in range(sequences.shape[1]):
        sequences[:,i] = ([0]*4**(l-i) + [1]*4**(l-i) + [2]*4**(l-i) +[3]*4**(l-i))*4**i
    while np.abs(Hprev - H) > cutoff:
        #print(np.abs(Hprev - H))
        Hprev = H
        for pos in range(sequences.shape[1]):
            for base in range(4):
                Q = np.sum(train[:,pos] == base)/float(train.shape[0])
                if Q == 0: continue
                Qhat = np.sum(np.exp(prob[sequences[:,pos] == base]))
                prob[sequences[:,pos] == base] += np.log(Q) - np.log(Qhat)
                prob[sequences[:,pos] != base] += np.log(1-Q) - np.log(1-Qhat)
                
                for pos2 in np.setdiff1d(range(sequences.shape[1]), range(pos+1)):
                    for base2 in range(4):
                        Q = np.sum((train[:,pos] == base)*(train[:,pos2] == base2))/float(train.shape[0])
                        if Q == 0: continue
                        which = (sequences[:,pos] == base)*(sequences[:,pos2] == base2)
                        Qhat = np.sum(np.exp(prob[which]))
                        prob[which] += np.log(Q) - np.log(Qhat)
                        prob[np.invert(which)] += np.log(1-Q) - np.log(1-Qhat)
                        
                        for pos3 in np.setdiff1d(range(sequences.shape[1]), range(pos2+1)):
                            for base3 in range(4):
                                Q = np.sum((train[:,pos] == base)*(train[:,pos2] == base2)*(train[:,pos3] == base3))/float(train.shape[0])
                                if Q == 0: continue
                                which = (sequences[:,pos] == base)*(sequences[:,pos2] == base2)*(sequences[:,pos3] == base3)
                                Qhat = np.sum(np.exp(prob[which]))
                                prob[which] += np.log(Q) - np.log(Qhat)
                                prob[np.invert(which)] += np.log(1-Q) - np.log(1-Qhat)
        H = -np.sum(prob*np.exp(prob))/np.log(2)
    return np.exp(prob)

def structuralParameters(genes, annotations, minIL = 0):
    # Get the empirical length distributions for introns and single, first, middle, and last exons, as well as number exons per gene
    
    # Transitions
    numExonsPerGene = [] 
    
    # Length Distributions
    lengthSingleExons = []
    lengthFirstExons = []
    lengthMiddleExons = []
    lengthLastExons = []
    lengthIntrons = []
    
    for gene in genes:
        if len(annotations[gene]) == 0: 
            print('missing annotation for', gene)
            continue
        numExons = 0
        introns = []
        singleExons = []
        firstExons = []
        middleExons = []
        lastExons = []
        
        for transcript in annotations[gene].values():
            numExons += len(transcript)
            
            # First exon 
            three = transcript[0][0] # Make three the first base
            five = transcript[0][1] + 1
            if len(transcript) == 1: 
                singleExons.append((three, five-1))
                continue # skip the rest for a single exon case
            firstExons.append((three, five-1)) # since three is the first base
            
            # Internal exons 
            for exon in transcript[1:-1]:
                three = exon[0] - 1 
                introns.append((five+1,three-1))
                five = exon[1] + 1
                middleExons.append((three+1, five-1))
                
            # Last exon 
            three = transcript[-1][0] - 1
            introns.append((five+1,three-1))
            five = transcript[-1][1] + 1
            lastExons.append((three+1, five-1))
        
        geneIntronLengths = [minIL]
        for intron in set(introns):
            geneIntronLengths.append(intron[1] - intron[0] + 1)
        
        if np.min(geneIntronLengths) < minIL: continue
        
        for intron in set(introns): lengthIntrons.append(intron[1] - intron[0] + 1)
        for exon in set(singleExons): lengthSingleExons.append(exon[1] - exon[0] + 1)
        for exon in set(firstExons): lengthFirstExons.append(exon[1] - exon[0] + 1)
        for exon in set(middleExons): lengthMiddleExons.append(exon[1] - exon[0] + 1)
        for exon in set(lastExons): lengthLastExons.append(exon[1] - exon[0] + 1)
            
        numExonsPerGene.append(float(numExons)/len(annotations[gene]))
        
    return(numExonsPerGene, lengthSingleExons, lengthFirstExons, lengthMiddleExons, lengthLastExons, lengthIntrons)

def adaptive_kde_tailed(lengths, N, geometric_cutoff = .8, lower_cutoff=0):
    adaptive_kde = GaussianKDE(alpha = 1) 
    adaptive_kde.fit(np.array(lengths)[:,None]) 
    
    lengths = np.array(lengths)
    join = np.sort(lengths)[int(len(lengths)*geometric_cutoff)] 
    
    smoothed = np.zeros(N)
    smoothed[:join+1] = adaptive_kde.predict(np.arange(join+1)[:,None])
    
    s = 1-np.sum(smoothed)
    p = smoothed[join]
    smoothed[join+1:] = np.exp(np.log(p) + np.log(s/(s+p))*np.arange(1,len(smoothed)-join))
    smoothed[:lower_cutoff] = 0
    smoothed /= np.sum(smoothed)
    
    return(smoothed)
    
def geometric_smooth_tailed(lengths, N, bandwidth, join, lower_cutoff=0):
    lengths = np.array(lengths)
    smoothing = KernelDensity(bandwidth = bandwidth).fit(lengths[:, np.newaxis]) 
    
    smoothed = np.zeros(N)
    smoothed[:join+1] = np.exp(smoothing.score_samples(np.arange(join+1)[:,None]))
    
    s = 1-np.sum(smoothed)
    p = smoothed[join]
    smoothed[join+1:] = np.exp(np.log(p) + np.log(s/(s+p))*np.arange(1,len(smoothed)-join))
    smoothed[:lower_cutoff] = 0
    smoothed /= np.sum(smoothed)
    
    return(smoothed)

def maxEnt5(geneNames, genes, dir):
    # Get all the 5'SS maxent scores for each of the genes in geneNames
    scores = {}
    prob = np.load(dir + '/maxEnt5_prob.npy')
    prob0 = np.load(dir + '/maxEnt5_prob0.npy') 
        
    for gene in geneNames:
        sequence = str(genes[gene].seq).lower()
        sequence5 = np.array([hashSequence(sequence[i:i+9]) for i in range(len(sequence)-9+1)])
        scores[gene] = np.zeros(len(sequence)) - np.inf
        scores[gene][3:-5] = np.log2(prob[sequence5]) - np.log2(prob0[sequence5])
        scores[gene] = np.exp2(scores[gene])
    
    return scores
    
def maxEnt5_single(seq, dir):
    # Get all the 5'SS maxent scores for the input sequence
    prob = np.load(dir + 'maxEnt5_prob.npy')
    prob0 = np.load(dir + 'maxEnt5_prob0.npy')
    
    seq = seq.lower()
    sequence5 = np.array([hashSequence(seq[i:i+9]) for i in range(len(seq)-9+1)])
    scores = np.log2(np.zeros(len(seq)))
    scores[3:-5] = np.log2(prob[sequence5]) - np.log2(prob0[sequence5])
    return np.exp2(scores)
    
def maxEnt3(geneNames, genes, dir):
    # Get all the 3'SS maxent scores for each of the genes in geneNames
    scores = {}
    prob0 = np.load(dir + 'maxEnt3_prob0.npy')
    prob1 = np.load(dir + 'maxEnt3_prob1.npy')
    prob2 = np.load(dir + 'maxEnt3_prob2.npy')
    prob3 = np.load(dir + 'maxEnt3_prob3.npy')
    prob4 = np.load(dir + 'maxEnt3_prob4.npy')
    prob5 = np.load(dir + 'maxEnt3_prob5.npy')
    prob6 = np.load(dir + 'maxEnt3_prob6.npy')
    prob7 = np.load(dir + 'maxEnt3_prob7.npy')
    prob8 = np.load(dir + 'maxEnt3_prob8.npy')
    
    prob0_0 = np.load(dir + 'maxEnt3_prob0_0.npy')
    prob1_0 = np.load(dir + 'maxEnt3_prob1_0.npy')
    prob2_0 = np.load(dir + 'maxEnt3_prob2_0.npy')
    prob3_0 = np.load(dir + 'maxEnt3_prob3_0.npy')
    prob4_0 = np.load(dir + 'maxEnt3_prob4_0.npy')
    prob5_0 = np.load(dir + 'maxEnt3_prob5_0.npy')
    prob6_0 = np.load(dir + 'maxEnt3_prob6_0.npy')
    prob7_0 = np.load(dir + 'maxEnt3_prob7_0.npy')
    prob8_0 = np.load(dir + 'maxEnt3_prob8_0.npy')
    
    for gene in geneNames:
        sequence = str(genes[gene].seq).lower()
        sequences23 = [sequence[i:i+23] for i in range(len(sequence)-23+1)]
        hash0 = np.array([hashSequence(seq[0:7]) for seq in sequences23])
        hash1 = np.array([hashSequence(seq[7:14]) for seq in sequences23])
        hash2 = np.array([hashSequence(seq[14:]) for seq in sequences23])
        hash3 = np.array([hashSequence(seq[4:11]) for seq in sequences23])
        hash4 = np.array([hashSequence(seq[11:18]) for seq in sequences23])
        hash5 = np.array([hashSequence(seq[4:7]) for seq in sequences23])
        hash6 = np.array([hashSequence(seq[7:11]) for seq in sequences23])
        hash7 = np.array([hashSequence(seq[11:14]) for seq in sequences23])
        hash8 = np.array([hashSequence(seq[14:18]) for seq in sequences23])
        
        probs = np.log2(prob0[hash0]) + np.log2(prob1[hash1]) + np.log2(prob2[hash2]) + \
            np.log2(prob3[hash3]) + np.log2(prob4[hash4]) - np.log2(prob5[hash5]) - \
            np.log2(prob6[hash6]) - np.log2(prob7[hash7]) - np.log2(prob8[hash8]) - \
            (np.log2(prob0_0[hash0]) + np.log2(prob1_0[hash1]) + np.log2(prob2_0[hash2]) + \
            np.log2(prob3_0[hash3]) + np.log2(prob4_0[hash4]) - np.log2(prob5_0[hash5]) - \
            np.log2(prob6_0[hash6]) - np.log2(prob7_0[hash7]) - np.log2(prob8_0[hash8]))
            
        scores[gene] = np.zeros(len(sequence)) - np.inf
        scores[gene][19:-3] = probs
        scores[gene] = np.exp2(scores[gene])
    
    return scores
    
def maxEnt3_single(seq, dir):
    # Get all the 3'SS maxent scores for the input sequence
    prob0 = np.load(dir + 'maxEnt3_prob0.npy')
    prob1 = np.load(dir + 'maxEnt3_prob1.npy')
    prob2 = np.load(dir + 'maxEnt3_prob2.npy')
    prob3 = np.load(dir + 'maxEnt3_prob3.npy')
    prob4 = np.load(dir + 'maxEnt3_prob4.npy')
    prob5 = np.load(dir + 'maxEnt3_prob5.npy')
    prob6 = np.load(dir + 'maxEnt3_prob6.npy')
    prob7 = np.load(dir + 'maxEnt3_prob7.npy')
    prob8 = np.load(dir + 'maxEnt3_prob8.npy')
    
    prob0_0 = np.load(dir + 'maxEnt3_prob0_0.npy')
    prob1_0 = np.load(dir + 'maxEnt3_prob1_0.npy')
    prob2_0 = np.load(dir + 'maxEnt3_prob2_0.npy')
    prob3_0 = np.load(dir + 'maxEnt3_prob3_0.npy')
    prob4_0 = np.load(dir + 'maxEnt3_prob4_0.npy')
    prob5_0 = np.load(dir + 'maxEnt3_prob5_0.npy')
    prob6_0 = np.load(dir + 'maxEnt3_prob6_0.npy')
    prob7_0 = np.load(dir + 'maxEnt3_prob7_0.npy')
    prob8_0 = np.load(dir + 'maxEnt3_prob8_0.npy')
    
    seq = seq.lower()
    sequences23 = [seq[i:i+23] for i in range(len(seq)-23+1)]
    hash0 = np.array([hashSequence(seq[0:7]) for seq in sequences23])
    hash1 = np.array([hashSequence(seq[7:14]) for seq in sequences23])
    hash2 = np.array([hashSequence(seq[14:]) for seq in sequences23])
    hash3 = np.array([hashSequence(seq[4:11]) for seq in sequences23])
    hash4 = np.array([hashSequence(seq[11:18]) for seq in sequences23])
    hash5 = np.array([hashSequence(seq[4:7]) for seq in sequences23])
    hash6 = np.array([hashSequence(seq[7:11]) for seq in sequences23])
    hash7 = np.array([hashSequence(seq[11:14]) for seq in sequences23])
    hash8 = np.array([hashSequence(seq[14:18]) for seq in sequences23])
    
    probs = np.log2(prob0[hash0]) + np.log2(prob1[hash1]) + np.log2(prob2[hash2]) + \
            np.log2(prob3[hash3]) + np.log2(prob4[hash4]) - np.log2(prob5[hash5]) - \
            np.log2(prob6[hash6]) - np.log2(prob7[hash7]) - np.log2(prob8[hash8]) - \
            (np.log2(prob0_0[hash0]) + np.log2(prob1_0[hash1]) + np.log2(prob2_0[hash2]) + \
            np.log2(prob3_0[hash3]) + np.log2(prob4_0[hash4]) - np.log2(prob5_0[hash5]) - \
            np.log2(prob6_0[hash6]) - np.log2(prob7_0[hash7]) - np.log2(prob8_0[hash8]))
            
    scores = np.log2(np.zeros(len(seq)))
    scores[19:-3] = probs
    return np.exp2(scores)

def sreScores_single(seq, sreScores, kmer = 6):
    indices = [hashSequence(seq[i:i+kmer]) for i in range(len(seq)-kmer+1)]
    sequenceSRES = [sreScores[indices[i]] for i in range(len(indices))]
    return sequenceSRES

def get_all_5ss(gene, reference, genes):
    # Get all the 5'SS for a gene based on the annotation in reference
    info = genes[gene].description.split(' ')
    if reference.loc[gene,6] == ',': exonEnds = []
    else: exonEnds = [int(start)-1 for start in reference.loc[gene,6].split(',')[:-1]] # intron starts -> exon ends
    if reference.loc[gene,7] == ',': exonStarts = []
    else: exonStarts = [int(end)+1 for end in reference.loc[gene,7].split(',')[:-1]] # exon starts -> intron ends
    
    if info[6] == 'Strand:-': 
        stop = int(info[5][5:])
        annnotation = [stop - exonStarts[i-1] + 2 for i in range(len(exonStarts),0,-1)]      
        
    elif info[6] == 'Strand:+': 
        start = int(info[4][6:])
        annnotation = [exonEnds[i] - start + 1 for i in range(len(exonEnds))]
        
    return(annnotation)

def get_all_3ss(gene, reference, genes):
    # Get all the 3'SS for a gene based on the annotation in reference
    info = genes[gene].description.split(' ')
    if reference.loc[gene,6] == ',': exonEnds = []
    else: exonEnds = [int(start)-1 for start in reference.loc[gene,6].split(',')[:-1]] # intron starts -> exon ends
    if reference.loc[gene,7] == ',': exonStarts = []
    else: exonStarts = [int(end)+1 for end in reference.loc[gene,7].split(',')[:-1]] # exon starts -> intron ends
    
    if info[6] == 'Strand:-': 
        stop = int(info[5][5:])
        annnotation = [stop - exonEnds[i-1] - 2 for i in range(len(exonEnds),0,-1)]      
        
    elif info[6] == 'Strand:+': 
        start = int(info[4][6:])
        annnotation = [exonStarts[i] - start - 3 for i in range(len(exonStarts))]
        
    return(annnotation)

def get_hexamer_real_decoy_counts(geneNames, trueSeqs, decoySS, genes, kmer, sreEffect5_exon, sreEffect5_intron, sreEffect3_exon, sreEffect3_intron, B3 = 3, B5 = 5):
    # Get the counts of hexamers in the flanking regions for real and decoy ss with restriction to exons and introns for the real ss
    true_counts_5_intron = np.zeros(4**kmer, dtype = np.dtype("i"))
    true_counts_5_exon = np.zeros(4**kmer, dtype = np.dtype("i"))
    true_counts_3_intron = np.zeros(4**kmer, dtype = np.dtype("i"))
    true_counts_3_exon = np.zeros(4**kmer, dtype = np.dtype("i"))
    decoy_counts_5_intron = np.zeros(4**kmer, dtype = np.dtype("i"))
    decoy_counts_5_exon = np.zeros(4**kmer, dtype = np.dtype("i"))
    decoy_counts_3_intron = np.zeros(4**kmer, dtype = np.dtype("i"))
    decoy_counts_3_exon = np.zeros(4**kmer, dtype = np.dtype("i"))
    
    for gene in geneNames:
        trueThrees = np.nonzero(trueSeqs[gene] == B3)[0][:-1]
        trueFives = np.nonzero(trueSeqs[gene] == B5)[0][1:]
        for i in range(len(trueThrees)):
            three = trueThrees[i]
            five = trueFives[i]
            
            # 3'SS
            sequence = str(genes[gene].seq[three+4:three+sreEffect3_exon+1].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[three+4:].lower())
            if five-3 < three+sreEffect3_exon+1: sequence = str(genes[gene].seq[three+4:five-3].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: true_counts_3_exon[s] += 1
            
            sequence = str(genes[gene].seq[three-sreEffect3_intron:three-19].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[:three-19].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: true_counts_3_intron[s] += 1
                
            # 5'SS
            sequence = str(genes[gene].seq[five-sreEffect5_exon:five-3].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[:five-3].lower())
            if five-sreEffect5_exon < three+4: sequence = str(genes[gene].seq[three+4:five-3].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: true_counts_5_exon[s] += 1
            
            sequence = str(genes[gene].seq[five+6:five+sreEffect5_intron+1].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[five+6:].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: true_counts_5_intron[s] += 1
        
        decoyThrees = np.nonzero(decoySS[gene] == B3)[0]
        decoyFives = np.nonzero(decoySS[gene] == B5)[0]
        for ss in decoyFives:
            sequence = str(genes[gene].seq[ss-sreEffect5_exon:ss-3].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[:ss-3].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: decoy_counts_5_exon[s] += 1
            
            sequence = str(genes[gene].seq[ss+6:ss+sreEffect5_intron+1].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[ss+6:].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: decoy_counts_5_intron[s] += 1
    
        for ss in decoyThrees:
            sequence = str(genes[gene].seq[ss+4:ss+sreEffect3_exon+1].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[ss+4:].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: decoy_counts_3_exon[s] += 1
            
            sequence = str(genes[gene].seq[ss-sreEffect3_intron:ss-19].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[:ss-19].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: decoy_counts_3_intron[s] += 1
    
    return(true_counts_5_intron, true_counts_5_exon, true_counts_3_intron, true_counts_3_exon, 
           decoy_counts_5_intron, decoy_counts_5_exon, decoy_counts_3_intron, decoy_counts_3_exon)

def get_hexamer_counts(geneNames, set1, set2, genes, kmer, sreEffect5_exon, sreEffect5_intron, sreEffect3_exon, sreEffect3_intron, B3 = 3, B5 = 5):
    # Get the counts of hexamers in the flanking regions for two sets of ss
    set1_counts_5_intron = np.zeros(4**kmer, dtype = np.dtype("i"))
    set1_counts_5_exon = np.zeros(4**kmer, dtype = np.dtype("i"))
    set1_counts_3_intron = np.zeros(4**kmer, dtype = np.dtype("i"))
    set1_counts_3_exon = np.zeros(4**kmer, dtype = np.dtype("i"))
    set2_counts_5_intron = np.zeros(4**kmer, dtype = np.dtype("i"))
    set2_counts_5_exon = np.zeros(4**kmer, dtype = np.dtype("i"))
    set2_counts_3_intron = np.zeros(4**kmer, dtype = np.dtype("i"))
    set2_counts_3_exon = np.zeros(4**kmer, dtype = np.dtype("i"))
    
    for gene in geneNames:
        set1Threes = np.nonzero(set1[gene] == B3)[0]
        set1Fives = np.nonzero(set1[gene] == B5)[0]
        set2Threes = np.nonzero(set2[gene] == B3)[0]
        set2Fives = np.nonzero(set2[gene] == B5)[0]
        
        for ss in set1Fives:
            sequence = str(genes[gene].seq[ss-sreEffect5_exon:ss-3].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[:ss-3].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: set1_counts_5_exon[s] += 1
            
            sequence = str(genes[gene].seq[ss+6:ss+sreEffect5_intron+1].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[ss+6:].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: set1_counts_5_intron[s] += 1
    
        for ss in set1Threes:
            sequence = str(genes[gene].seq[ss+4:ss+sreEffect3_exon+1].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[ss+4:].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: set1_counts_3_exon[s] += 1
            
            sequence = str(genes[gene].seq[ss-sreEffect3_intron:ss-19].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[:ss-19].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: set1_counts_3_intron[s] += 1
        
        for ss in set2Fives:
            sequence = str(genes[gene].seq[ss-sreEffect5_exon:ss-3].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[:ss-3].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: set2_counts_5_exon[s] += 1
            
            sequence = str(genes[gene].seq[ss+6:ss+sreEffect5_intron+1].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[ss+6:].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: set2_counts_5_intron[s] += 1
    
        for ss in set2Threes:
            sequence = str(genes[gene].seq[ss+4:ss+sreEffect3_exon+1].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[ss+4:].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: set2_counts_3_exon[s] += 1
            
            sequence = str(genes[gene].seq[ss-sreEffect3_intron:ss-19].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[:ss-19].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: set2_counts_3_intron[s] += 1
    
    return(set1_counts_5_intron, set1_counts_5_exon, set1_counts_3_intron, set1_counts_3_exon, 
           set2_counts_5_intron, set2_counts_5_exon, set2_counts_3_intron, set2_counts_3_exon)

def get_hexamer_real_decoy_scores(geneNames, trueSeqs, decoySS, genes, kmer, sreEffect5_exon, sreEffect5_intron, sreEffect3_exon, sreEffect3_intron):
    # Get the real versus decoy scores for all hexamers
    true_counts_5_intron, true_counts_5_exon, true_counts_3_intron, true_counts_3_exon, decoy_counts_5_intron, decoy_counts_5_exon, decoy_counts_3_intron, decoy_counts_3_exon = get_hexamer_real_decoy_counts(geneNames, trueSeqs, decoySS, genes, kmer = kmer, sreEffect5_exon = sreEffect5_exon, sreEffect5_intron = sreEffect5_intron, sreEffect3_exon = sreEffect3_exon, sreEffect3_intron = sreEffect3_intron)
    
    # Add pseudocounts
    true_counts_5_intron = true_counts_5_intron + 1
    true_counts_5_exon = true_counts_5_exon + 1
    true_counts_3_intron = true_counts_3_intron + 1
    true_counts_3_exon = true_counts_3_exon + 1
    decoy_counts_5_intron = decoy_counts_5_intron + 1
    decoy_counts_5_exon = decoy_counts_5_exon + 1
    decoy_counts_3_intron = decoy_counts_3_intron + 1
    decoy_counts_3_exon = decoy_counts_3_exon + 1
    
    true_counts_intron = true_counts_5_intron + true_counts_3_intron
    true_counts_exon = true_counts_5_exon + true_counts_3_exon
    decoy_counts_intron = decoy_counts_5_intron + decoy_counts_3_intron
    decoy_counts_exon = decoy_counts_5_exon + decoy_counts_3_exon
    
    trueFreqs_intron = np.exp(np.log(true_counts_intron) - np.log(np.sum(true_counts_intron))) 
    decoyFreqs_intron = np.exp(np.log(decoy_counts_intron) - np.log(np.sum(decoy_counts_intron)))
    trueFreqs_exon = np.exp(np.log(true_counts_exon) - np.log(np.sum(true_counts_exon)))
    decoyFreqs_exon = np.exp(np.log(decoy_counts_exon) - np.log(np.sum(true_counts_exon)))
    
    sreScores_intron = np.exp(np.log(true_counts_intron) - np.log(np.sum(true_counts_intron)) 
                              - np.log(decoy_counts_intron) + np.log(np.sum(decoy_counts_intron)))
    sreScores_exon = np.exp(np.log(true_counts_exon) - np.log(np.sum(true_counts_exon)) 
                            - np.log(decoy_counts_exon) + np.log(np.sum(decoy_counts_exon)))
    
    sreScores3_intron = np.exp(np.log(true_counts_3_intron) - np.log(np.sum(true_counts_3_intron)) 
                                - np.log(decoy_counts_3_intron) + np.log(np.sum(decoy_counts_3_intron)))
    sreScores3_exon = np.exp(np.log(true_counts_3_exon) - np.log(np.sum(true_counts_3_exon)) 
                              - np.log(decoy_counts_3_exon) + np.log(np.sum(decoy_counts_3_exon)))
    
    sreScores5_intron = np.exp(np.log(true_counts_5_intron) - np.log(np.sum(true_counts_5_intron)) 
                                - np.log(decoy_counts_5_intron) + np.log(np.sum(decoy_counts_5_intron)))
    sreScores5_exon = np.exp(np.log(true_counts_5_exon) - np.log(np.sum(true_counts_5_exon)) 
                              - np.log(decoy_counts_5_exon) + np.log(np.sum(decoy_counts_5_exon)))
    
    return(sreScores_intron, sreScores_exon, sreScores3_intron, sreScores3_exon, sreScores5_intron, sreScores5_exon)
    
def score_sequences(sequences, exonicSREs5s, exonicSREs3s, intronicSREs5s, intronicSREs3s, k = 6, sreEffect5_exon = 80, sreEffect5_intron = 80, sreEffect3_exon = 80, sreEffect3_intron = 80, meDir = ''): 
    # Get the CASS scores for the input sequences
    
    batch_size = len(sequences)
    lengths = np.zeros(batch_size, dtype=np.dtype("i"))
    
    # Collect the lengths of each sequence in the batch
    for g in range(batch_size): 
        lengths[g] = len(sequences[g])
    L = np.max(lengths)
    
    emissions3 = np.log(np.zeros((batch_size, L), dtype=np.dtype("d")))
    emissions5 = np.log(np.zeros((batch_size, L), dtype=np.dtype("d")))
    
    # Get the emissions and apply sre scores to them
    for g in range(batch_size): 
        # 5'SS exonic effects (upstream)
        ssRange = 3
        emissions5[g,:lengths[g]] = np.log(maxEnt5_single(sequences[g].lower(), meDir))
        emissions5[g,k+ssRange:lengths[g]] += np.cumsum(exonicSREs5s[g,:lengths[g]-k+1])[:-1-ssRange]
        emissions5[g,sreEffect5_exon+1:lengths[g]] -= np.cumsum(exonicSREs5s[g,:lengths[g]-k+1])[:-(sreEffect5_exon+1)+(k-1)]
        
        # 3'SS intronic effects (upstream)
        ssRange = 19
        emissions3[g,:lengths[g]] = np.log(maxEnt3_single(sequences[g].lower(), meDir))
        emissions3[g,k+ssRange:lengths[g]] += np.cumsum(intronicSREs3s[g,:lengths[g]-k+1])[:-1-ssRange]
        emissions3[g,sreEffect3_intron+1:lengths[g]] -= np.cumsum(intronicSREs3s[g,:lengths[g]-k+1])[:-(sreEffect3_intron+1)+(k-1)]
        
        # 5'SS intronic effects (downstream)
        ssRange = 4
        emissions5[g,:lengths[g]-sreEffect5_intron] += np.cumsum(intronicSREs5s[g,:lengths[g]-k+1])[sreEffect5_intron-k+1:]
        emissions5[g,lengths[g]-sreEffect5_intron:lengths[g]-k+1-ssRange] += np.sum(intronicSREs5s[g,:lengths[g]-k+1])
        emissions5[g,:lengths[g]-k+1-ssRange] -= np.cumsum(intronicSREs5s[g,ssRange:lengths[g]-k+1])
        
        # 3'SS exonic effects (downstream)
        ssRange = 3
        emissions3[g,:lengths[g]-sreEffect5_exon] += np.cumsum(exonicSREs3s[g,:lengths[g]-k+1])[sreEffect5_exon-k+1:]
        emissions3[g,lengths[g]-sreEffect5_exon:lengths[g]-k+1-ssRange] += np.sum(exonicSREs3s[g,:lengths[g]-k+1])
        emissions3[g,:lengths[g]-k+1-ssRange] -= np.cumsum(exonicSREs3s[g,ssRange:lengths[g]-k+1])
        
    return np.exp(emissions5), np.exp(emissions3)

def cass_accuracy_metrics(scored_sequences_5, scored_sequences_3, geneNames, trueSeqs, B3 = 3, B5 = 5):
    # Get the best cutoff and the associated metrics for the CASS scored sequences
    true_scores = []
    for g, gene in enumerate(geneNames):
        L = len(trueSeqs[gene])
        for score in np.log2(scored_sequences_5[g,:L][trueSeqs[gene] == B5]): true_scores.append(score)
        for score in np.log2(scored_sequences_3[g,:L][trueSeqs[gene] == B3]): true_scores.append(score)
    min_score = np.min(true_scores)
    if np.isnan(min_score): 
        return 0, 0, 0, min_score
    
    all_scores = []
    for g, gene in enumerate(geneNames):
        L = len(trueSeqs[gene])
        for score in np.log2(scored_sequences_5[g,:L][trueSeqs[gene] != B5]): 
            if score > min_score: all_scores.append(score)
        for score in np.log2(scored_sequences_3[g,:L][trueSeqs[gene] != B3]): 
            if score > min_score: all_scores.append(score)
    
    all_scores = np.array(true_scores + all_scores)
    all_scores_bool = np.zeros(len(all_scores), dtype=np.dtype("i"))
    all_scores_bool[:len(true_scores)] = 1
    sort_inds = np.argsort(all_scores)
    all_scores = all_scores[sort_inds]
    all_scores_bool = all_scores_bool[sort_inds]
    
    num_all_positives = len(true_scores)
    num_all = len(all_scores)
    best_f1 = 0
    best_cutoff = 0
    for i, cutoff in enumerate(all_scores):
        if all_scores_bool[i] == 0: continue
        true_positives = np.sum(all_scores_bool[i:])
        false_negatives = num_all_positives - true_positives
        false_positives = num_all - i - true_positives
        
        ssSens = true_positives / (true_positives + false_negatives)
        ssPrec = true_positives / (true_positives + false_positives)
        f1 = 2 / (1/ssSens + 1/ssPrec)
        if f1 >= best_f1:
            best_f1 = f1
            best_cutoff = cutoff
            best_sens = ssSens
            best_prec = ssPrec
        
    return best_sens, best_prec, best_f1, best_cutoff
    
def cass_accuracy_metrics_set_cutoff(scored_sequences_5, scored_sequences_3, geneNames, trueSeqs, cutoff, B3 = 3, B5 = 5):
    # Get the associated metrics for the CASS scored sequences with a given cutoff
    true_scores = []
    for g, gene in enumerate(geneNames):
        L = len(trueSeqs[gene])
        for score in np.log2(scored_sequences_5[g,:L][trueSeqs[gene] == B5]): true_scores.append(score)
        for score in np.log2(scored_sequences_3[g,:L][trueSeqs[gene] == B3]): true_scores.append(score)
    min_score = np.min(true_scores)
    if np.isnan(min_score): 
        return 0, 0, 0
    
    all_scores = []
    for g, gene in enumerate(geneNames):
        L = len(trueSeqs[gene])
        for score in np.log2(scored_sequences_5[g,:L][trueSeqs[gene] != B5]): 
            if score > min_score: all_scores.append(score)
        for score in np.log2(scored_sequences_3[g,:L][trueSeqs[gene] != B3]): 
            if score > min_score: all_scores.append(score)
    
    all_scores = np.array(true_scores + all_scores)
    all_scores_bool = np.zeros(len(all_scores), dtype=np.dtype("i"))
    all_scores_bool[:len(true_scores)] = 1
    sort_inds = np.argsort(all_scores)
    all_scores = all_scores[sort_inds]
    all_scores_bool = all_scores_bool[sort_inds]
    
    num_all_positives = len(true_scores)
    num_all = len(all_scores)
    
    true_positives = np.sum((all_scores > cutoff)&(all_scores_bool == 1))
    false_negatives = num_all_positives - true_positives
    false_positives = np.sum((all_scores > cutoff)&(all_scores_bool == 0))
    
    ssSens = true_positives / (true_positives + false_negatives)
    ssPrec = true_positives / (true_positives + false_positives)
    f1 = 2 / (1/ssSens + 1/ssPrec)
        
    return ssSens, ssPrec, f1

def viterbi(sequences, transitions, pIL, pELS, pELF, pELM, pELL, exonicSREs5s, exonicSREs3s, intronicSREs5s, intronicSREs3s, k, sreEffect5_exon, sreEffect5_intron, sreEffect3_exon, sreEffect3_intron, meDir = ''): 
    # Get the best parses of all the input sequences
    
    batch_size = len(sequences)
    tbindex = np.zeros(batch_size, dtype=np.dtype("i"))
    lengths = np.zeros(batch_size, dtype=np.dtype("i"))
    loglik = np.log(np.zeros(batch_size, dtype=np.dtype("d")))
    
    # Collect the lengths of each sequence in the batch
    for g in range(batch_size): 
        lengths[g] = len(sequences[g])
    L = np.max(lengths)
    
    emissions3 = np.log(np.zeros((batch_size, L), dtype=np.dtype("d")))
    emissions5 = np.log(np.zeros((batch_size, L), dtype=np.dtype("d"))) 
    Three = np.log(np.zeros((batch_size, L), dtype=np.dtype("d")))    
    Five = np.log(np.zeros((batch_size, L), dtype=np.dtype("d")))
    traceback5 = np.zeros((batch_size, L), dtype=np.dtype("i")) + L
    traceback3 = np.zeros((batch_size, L), dtype=np.dtype("i")) + L
    bestPath = np.zeros((batch_size, L), dtype=np.dtype("i"))
    
    # Rewind state vars
    exon = 2
    intron = 1
     
    # Convert inputs to log space
    transitions = np.log(transitions)
    pIL = np.log(pIL)
    pELS = np.log(pELS)
    pELF = np.log(pELF)
    pELM = np.log(pELM)
    pELL = np.log(pELL)
    
    # Get the emissions and apply sre scores to them
    for g in range(batch_size): 
        # 5'SS exonic effects (upstream)
        ssRange = 3
        emissions5[g,:lengths[g]] = np.log(maxEnt5_single(sequences[g].lower(), meDir))
        emissions5[g,k+ssRange:lengths[g]] += np.cumsum(exonicSREs5s[g,:lengths[g]-k+1])[:-1-ssRange]
        emissions5[g,sreEffect5_exon+1:lengths[g]] -= np.cumsum(exonicSREs5s[g,:lengths[g]-k+1])[:-(sreEffect5_exon+1)+(k-1)]
        
        # 3'SS intronic effects (upstream)
        ssRange = 19
        emissions3[g,:lengths[g]] = np.log(maxEnt3_single(sequences[g].lower(), meDir))
        emissions3[g,k+ssRange:lengths[g]] += np.cumsum(intronicSREs3s[g,:lengths[g]-k+1])[:-1-ssRange]
        emissions3[g,sreEffect3_intron+1:lengths[g]] -= np.cumsum(intronicSREs3s[g,:lengths[g]-k+1])[:-(sreEffect3_intron+1)+(k-1)]
        
        # 5'SS intronic effects (downstream)
        ssRange = 4
        emissions5[g,:lengths[g]-sreEffect5_intron] += np.cumsum(intronicSREs5s[g,:lengths[g]-k+1])[sreEffect5_intron-k+1:]
        emissions5[g,lengths[g]-sreEffect5_intron:lengths[g]-k+1-ssRange] += np.sum(intronicSREs5s[g,:lengths[g]-k+1])
        emissions5[g,:lengths[g]-k+1-ssRange] -= np.cumsum(intronicSREs5s[g,ssRange:lengths[g]-k+1])
        
        # 3'SS exonic effects (downstream)
        ssRange = 3
        emissions3[g,:lengths[g]-sreEffect5_exon] += np.cumsum(exonicSREs3s[g,:lengths[g]-k+1])[sreEffect5_exon-k+1:]
        emissions3[g,lengths[g]-sreEffect5_exon:lengths[g]-k+1-ssRange] += np.sum(exonicSREs3s[g,:lengths[g]-k+1])
        emissions3[g,:lengths[g]-k+1-ssRange] -= np.cumsum(exonicSREs3s[g,ssRange:lengths[g]-k+1])
    
    # Convert the transition vector into named probabilities
    pME = transitions[0]
    p1E = np.log(1 - np.exp(pME))
    pEE = transitions[1]
    pEO = np.log(1 - np.exp(pEE))
    
    # Initialize the first and single exon probabilities
    ES = np.zeros(batch_size, dtype=np.dtype("d"))
    for g in range(batch_size): ES[g] = pELS[L-1] + p1E
    
    for g in range(batch_size): # loop the sequences in the batch
        for t in range(1,lengths[g]):
            Five[g,t] = pELF[t-1]
            
            for d in range(t,0,-1):
                # 5'SS
                if pEE + Three[g,t-d-1] + pELM[d-1] > Five[g,t]:
                    traceback5[g,t] = d
                    Five[g,t] = pEE + Three[g,t-d-1] + pELM[d-1]
            
                # 3'SS
                if Five[g,t-d-1] + pIL[d-1] > Three[g,t]:
                    traceback3[g,t] = d
                    Three[g,t] = Five[g,t-d-1] + pIL[d-1]
                    
            Five[g,t] += emissions5[g,t]
            Three[g,t] += emissions3[g,t]
            
        for i in range(1, lengths[g]):
            if pME + Three[g,i] + pEO + pELL[lengths[g]-i-2] > loglik[g]:
                loglik[g] = pME + Three[g,i] + pEO + pELL[lengths[g]-i-2]
                tbindex[g] = i
                
        if ES[g] <= loglik[g]: # If the single exon case isn't better, trace back
            while 0 < tbindex[g]:
                bestPath[g,tbindex[g]] = 3
                tbindex[g] -= traceback3[g,tbindex[g]] + 1
                bestPath[g,tbindex[g]] = 5
                tbindex[g] -= traceback5[g,tbindex[g]] + 1 
        else:
            loglik[g] = ES[g]
        
    return bestPath, loglik, emissions5, emissions3

def viterbi_intron(sequences, pIO, pIL, pELM, exonicSREs5s, exonicSREs3s, intronicSREs5s, intronicSREs3s, k, sreEffect5_exon, sreEffect5_intron, sreEffect3_exon, sreEffect3_intron, meDir = ''): 
    # Get the best parses of all the input sequences
    
    batch_size = len(sequences)
    tbindex = np.zeros(batch_size, dtype=np.dtype("i"))
    lengths = np.zeros(batch_size, dtype=np.dtype("i"))
    loglik = np.log(np.zeros(batch_size, dtype=np.dtype("d")))
    
    # Collect the lengths of each sequence in the batch
    for g in range(batch_size): 
        lengths[g] = len(sequences[g])
    L = np.max(lengths)
    
    emissions3 = np.log(np.zeros((batch_size, L), dtype=np.dtype("d")))
    emissions5 = np.log(np.zeros((batch_size, L), dtype=np.dtype("d"))) 
    Three = np.log(np.zeros((batch_size, L), dtype=np.dtype("d")))    
    Five = np.log(np.zeros((batch_size, L), dtype=np.dtype("d")))
    traceback5 = np.zeros((batch_size, L), dtype=np.dtype("i")) + L
    traceback3 = np.zeros((batch_size, L), dtype=np.dtype("i")) + L
    bestPath = np.zeros((batch_size, L), dtype=np.dtype("i"))
    
    # Rewind state vars
    exon = 2
    intron = 1
     
    # Convert inputs to log space
    pIO = np.log(pIO)
    pEE = np.log(1 - np.exp(pIO))
    pIL = np.log(pIL)
    pELM = np.log(pELM)
    
    # Get the emissions and apply sre scores to them
    for g in range(batch_size): 
        # 5'SS exonic effects (upstream)
        ssRange = 3
        emissions5[g,:lengths[g]] = np.log(maxEnt5_single(sequences[g].lower(), meDir))
        emissions5[g,k+ssRange:lengths[g]] += np.cumsum(exonicSREs5s[g,:lengths[g]-k+1])[:-1-ssRange]
        emissions5[g,sreEffect5_exon+1:lengths[g]] -= np.cumsum(exonicSREs5s[g,:lengths[g]-k+1])[:-(sreEffect5_exon+1)+(k-1)]
        
        # 3'SS intronic effects (upstream)
        ssRange = 19
        emissions3[g,:lengths[g]] = np.log(maxEnt3_single(sequences[g].lower(), meDir))
        emissions3[g,k+ssRange:lengths[g]] += np.cumsum(intronicSREs3s[g,:lengths[g]-k+1])[:-1-ssRange]
        emissions3[g,sreEffect3_intron+1:lengths[g]] -= np.cumsum(intronicSREs3s[g,:lengths[g]-k+1])[:-(sreEffect3_intron+1)+(k-1)]
        
        # 5'SS intronic effects (downstream)
        ssRange = 4
        emissions5[g,:lengths[g]-sreEffect5_intron] += np.cumsum(intronicSREs5s[g,:lengths[g]-k+1])[sreEffect5_intron-k+1:]
        emissions5[g,lengths[g]-sreEffect5_intron:lengths[g]-k+1-ssRange] += np.sum(intronicSREs5s[g,:lengths[g]-k+1])
        emissions5[g,:lengths[g]-k+1-ssRange] -= np.cumsum(intronicSREs5s[g,ssRange:lengths[g]-k+1])
        
        # 3'SS exonic effects (downstream)
        ssRange = 3
        emissions3[g,:lengths[g]-sreEffect5_exon] += np.cumsum(exonicSREs3s[g,:lengths[g]-k+1])[sreEffect5_exon-k+1:]
        emissions3[g,lengths[g]-sreEffect5_exon:lengths[g]-k+1-ssRange] += np.sum(exonicSREs3s[g,:lengths[g]-k+1])
        emissions3[g,:lengths[g]-k+1-ssRange] -= np.cumsum(exonicSREs3s[g,ssRange:lengths[g]-k+1])
    
    
    # Initialize the first and single exon probabilities
    IS = np.zeros(batch_size, dtype=np.dtype("d"))
    for g in range(batch_size): IS[g] = pIL[L-1] + pIO
    
    for g in range(batch_size): # loop the sequences in the batch
        for t in range(1,lengths[g]):
            Three[g,t] = pIL[t-1] + pEE
            
            for d in range(t,0,-1):
                # 5'SS
                if Three[g,t-d-1] + pELM[d-1] > Five[g,t]:
                    traceback5[g,t] = d
                    Five[g,t] = Three[g,t-d-1] + pELM[d-1]
            
                # 3'SS
                if Five[g,t-d-1] + pIL[d-1] > Three[g,t]:
                    traceback3[g,t] = d
                    Three[g,t] = Five[g,t-d-1] + pIL[d-1]
                    
            Five[g,t] += emissions5[g,t]
            Three[g,t] += emissions3[g,t]
            
        for i in range(1, lengths[g]):
            if Five[g,i] + pIO + pIL[lengths[g]-i-2] > loglik[g]:
                loglik[g] = Five[g,i] + pIO + pIL[lengths[g]-i-2]
                tbindex[g] = i
                
        if IS[g] <= loglik[g]: # If the single intron case isn't better, trace back
            while 0 < tbindex[g]:
                bestPath[g,tbindex[g]] = 5
                tbindex[g] -= traceback5[g,tbindex[g]] + 1 
                bestPath[g,tbindex[g]] = 3
                tbindex[g] -= traceback3[g,tbindex[g]] + 1
        else:
            loglik[g] = IS[g]
        
    return bestPath, loglik, emissions5, emissions3

