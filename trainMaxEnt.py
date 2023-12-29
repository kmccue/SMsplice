import numpy as np
import pandas as pd
import time, argparse#, json, pickle
from Bio import SeqIO, SeqUtils, motifs
from Bio.Seq import Seq 
from Bio.SeqRecord import SeqRecord
import scipy.ndimage
import scipy.stats as stats
from SMsplice import *

startTime = time.time()

def parse_arguments():
    #argument parsing 
    parser = argparse.ArgumentParser(description="parsing arguments")
    parser.add_argument("-a", "--all_ss", required=True)
    parser.add_argument("-g", "--genome", required=True)
    parser.add_argument("-m", "--maxent_dir", required=True)
    
    opts = parser.parse_args()
    return opts

args = parse_arguments()

# Load data from arguments
maxEntDir = args.maxent_dir
genome = SeqIO.to_dict(SeqIO.parse(args.genome, "fasta"))
allSS = pd.read_csv(args.all_ss, sep = '\t', engine='python', index_col=0, header = None) 
allSS.index = allSS.index.map(str)

genes = {}
for gene in allSS.index:
    txstart = allSS.loc[gene,4]-1
    txend = allSS.loc[gene,5]
    chrom = allSS.loc[gene,2][3:]
    if not np.array_equiv(['a', 'c', 'g', 't'],
         np.unique(genome[chrom][allSS.loc[gene,4]-1:allSS.loc[gene,5]].seq.lower())): 
#         print(gene, "nonstandard base")
        continue
    
    name = gene
    geneID = gene
    description = gene + ' GeneID:'+gene + ' TranscriptID:Canonical' + ' Chromosome:' + allSS.loc[gene,2] + \
    ' Start:'+str(txstart) + ' Stop:'+str(txend) + ' Strand:'+allSS.loc[gene,3]

    if allSS.loc[gene,3] == '-':
        seq = genome[chrom][allSS.loc[gene,4]-1:allSS.loc[gene,5]].seq.reverse_complement()
    elif allSS.loc[gene,3] == '+': 
        seq = genome[chrom][allSS.loc[gene,4]-1:allSS.loc[gene,5]].seq
    else:
        print(gene, "strand error")
    
    genes[gene] = SeqRecord(seq, name = name, id = geneID, description = description)

# Additional parameters
sreEffect = 80
sreEffect3_intron = sreEffect + 19
sreEffect3_exon = sreEffect + 3
sreEffect5_intron = sreEffect + 5
sreEffect5_exon = sreEffect + 3

kmer = 6
E = 0
I = 1
B5 = 5
B3 = 3
train_size = 4000
test_size = 1000
score_learning_rate = .01

# Get training, validation, generalization, and test sets
testGenes = allSS[(allSS[1] == 0)&allSS[2].isin(['chr1', 'chr3', 'chr5', 'chr9'])].index 
testGenes = np.intersect1d(testGenes, list(genes.keys()))

trainGenes = allSS.index
trainGenes = np.intersect1d(trainGenes, list(genes.keys()))
trainGenes = np.setdiff1d(trainGenes, testGenes)

lengthsOfGenes = np.array([len(str(genes[gene].seq)) for gene in trainGenes])
trainGenes = trainGenes[lengthsOfGenes > sreEffect3_intron]
lengthsOfGenes = np.array([len(str(genes[gene].seq)) for gene in trainGenes])
validationGenes = trainGenes[lengthsOfGenes < 200000]

generalizationGenes = np.intersect1d(validationGenes, allSS[allSS[1] == 0].index)
if len(generalizationGenes) > test_size: generalizationGenes = np.random.choice(generalizationGenes, test_size, replace = False)
validationGenes = np.setdiff1d(validationGenes, generalizationGenes)
if len(validationGenes) > test_size: validationGenes = np.random.choice(validationGenes, test_size, replace = False)

trainGenes = np.setdiff1d(trainGenes, generalizationGenes)
trainGenes = np.setdiff1d(trainGenes, validationGenes)
if len(trainGenes) > train_size: trainGenes = np.random.choice(trainGenes, train_size, replace = False)

# 5'SS Training Sets
true = open(maxEntDir + '/train5_true.txt', 'w')
null = open(maxEntDir + '/train5_null.txt', 'w')

for gene in allSS.index.values:
    if gene not in genes.keys(): continue
    
    info = genes[gene].description.split(' ')
    seq = genes[gene].seq.lower()
    
    if gene in testGenes: continue
    if not np.array_equiv(['a', 'c', 'g', 't'], np.unique(seq)): continue
    if len(seq) < 300: continue
    
    exonEnds = [int(start)-1 for start in allSS.loc[gene,6].split(',')[:-1]] # intron starts -> exon ends
    exonStarts = [int(end)+1 for end in allSS.loc[gene,7].split(',')[:-1]] # exon starts -> intron ends
    
    if info[6] == 'Strand:-': 
        stop = int(info[5][5:])
        SSs = [stop - exonStarts[i-1] + 2 for i in range(len(exonStarts),0,-1)]    
        for ss in SSs: true.write(str(seq[ss-3:ss+6]) + "\n")
        
        notSS = np.setdiff1d(range(100 + 2, len(seq) - 200 - 10), SSs)
        for notss in np.random.choice(notSS, min(20*len(SSs), len(notSS)), replace = False):
            notssSeq = seq[notss-3:notss+6]
            if len(np.setdiff1d(list(set(notssSeq)), ['a', 'c', 'g', 't'])) == 0: 
                null.write(str(notssSeq) + "\n") 
        
    elif info[6] == 'Strand:+': 
        start = int(info[4][6:])
        SSs = [exonEnds[i] - start + 1 for i in range(len(exonEnds))]
        for ss in SSs: true.write(str(seq[ss-3:ss+6]) + "\n")
            
        notSS = np.setdiff1d(range(100 + 2, len(seq) - 200 - 10), SSs)
        for notss in np.random.choice(notSS, min(20*len(SSs), len(notSS)), replace = False):
            notssSeq = seq[notss-3:notss+6]
            if len(np.setdiff1d(list(set(notssSeq)), ['a', 'c', 'g', 't'])) == 0: 
                null.write(str(notssSeq) + "\n") 

true.close() 
null.close() 

# 3'SS Training Sets
true = open(maxEntDir + '/train3_true.txt', 'w')
null = open(maxEntDir + '/train3_null.txt', 'w')

for gene in trainGenes:
    if gene not in genes.keys(): continue
    
    info = genes[gene].description.split(' ')
    seq = genes[gene].seq.lower()
    
    if gene in testGenes: continue
    if not np.array_equiv(['a', 'c', 'g', 't'], np.unique(seq)): continue   
    if len(seq) < 300: continue
    
    exonEnds = [int(start)-1 for start in allSS.loc[gene,6].split(',')[:-1]] # intron starts -> exon ends
    exonStarts = [int(end)+1 for end in allSS.loc[gene,7].split(',')[:-1]] # exon starts -> intron ends
    
    if info[6] == 'Strand:-': 
        stop = int(info[5][5:])
        SSs = [stop - exonEnds[i-1] - 2 for i in range(len(exonEnds),0,-1)]
        for ss in SSs: true.write(str(seq[ss-19:ss+4]) + "\n")
            
        notSS = np.setdiff1d(range(100 + 20, len(seq) - 200 - 3), SSs)
        for notss in np.random.choice(notSS, min(20*len(SSs), len(notSS)), replace = False):
            notssSeq = seq[notss-19:notss+4]
            if len(np.setdiff1d(list(set(notssSeq)), ['a', 'c', 'g', 't'])) == 0: 
                null.write(str(notssSeq) + "\n") 
        
    elif info[6] == 'Strand:+': 
        start = int(info[4][6:])
        SSs = [exonStarts[i] - start - 3 for i in range(len(exonStarts))]
        for ss in SSs: true.write(str(seq[ss-19:ss+4]) + "\n")
            
        notSS = np.setdiff1d(range(100 + 20, len(seq) - 200 - 3), SSs)
        for notss in np.random.choice(notSS, min(20*len(SSs), len(notSS)), replace = False):
            notssSeq = seq[notss-19:notss+4]
            if len(np.setdiff1d(list(set(notssSeq)), ['a', 'c', 'g', 't'])) == 0: 
                null.write(str(notssSeq) + "\n") 

true.close() 
null.close() 

# 5'SS Training
with open(maxEntDir + '/train5_true.txt') as f: true_train = np.array(f.read().splitlines())
with open(maxEntDir + '/train5_null.txt') as f: null_train = np.array(f.read().splitlines())
prob = trainAllTriplets(true_train) 
prob_0 = trainAllTriplets(null_train) 
np.save(maxEntDir + '/maxEnt5_prob', prob)
np.save(maxEntDir + '/maxEnt5_prob0', prob_0)

# 3'SS Training
with open(maxEntDir + '/train3_true.txt') as f: true_train = np.array(f.read().splitlines())
prob0 = trainAllTriplets(np.array([seq[0:7] for seq in true_train]))
prob1 = trainAllTriplets(np.array([seq[7:14] for seq in true_train]))
prob2 = trainAllTriplets(np.array([seq[14:] for seq in true_train]))
prob3 = trainAllTriplets(np.array([seq[4:11] for seq in true_train]))
prob4 = trainAllTriplets(np.array([seq[11:18] for seq in true_train]))
prob5 = trainAllTriplets(np.array([seq[4:7] for seq in true_train]))
prob6 = trainAllTriplets(np.array([seq[7:11] for seq in true_train]))
prob7 = trainAllTriplets(np.array([seq[11:14] for seq in true_train]))
prob8 = trainAllTriplets(np.array([seq[14:18] for seq in true_train])) 

with open(maxEntDir + '/train3_null.txt') as f: null_train = np.array(f.read().splitlines())
prob0_0 = trainAllTriplets(np.array([seq[0:7] for seq in null_train]))
prob1_0 = trainAllTriplets(np.array([seq[7:14] for seq in null_train]))
prob2_0 = trainAllTriplets(np.array([seq[14:] for seq in null_train]))
prob3_0 = trainAllTriplets(np.array([seq[4:11] for seq in null_train]))
prob4_0 = trainAllTriplets(np.array([seq[11:18] for seq in null_train]))
prob5_0 = trainAllTriplets(np.array([seq[4:7] for seq in null_train]))
prob6_0 = trainAllTriplets(np.array([seq[7:11] for seq in null_train]))
prob7_0 = trainAllTriplets(np.array([seq[11:14] for seq in null_train]))
prob8_0 = trainAllTriplets(np.array([seq[14:18] for seq in null_train]))

np.save(maxEntDir + '/maxEnt3_prob0', prob0)
np.save(maxEntDir + '/maxEnt3_prob0_0', prob0_0)
np.save(maxEntDir + '/maxEnt3_prob1', prob1)
np.save(maxEntDir + '/maxEnt3_prob1_0', prob1_0)
np.save(maxEntDir + '/maxEnt3_prob2', prob2)
np.save(maxEntDir + '/maxEnt3_prob2_0', prob2_0)
np.save(maxEntDir + '/maxEnt3_prob3', prob3)
np.save(maxEntDir + '/maxEnt3_prob3_0', prob3_0)
np.save(maxEntDir + '/maxEnt3_prob4', prob4)
np.save(maxEntDir + '/maxEnt3_prob4_0', prob4_0)
np.save(maxEntDir + '/maxEnt3_prob5', prob5)
np.save(maxEntDir + '/maxEnt3_prob5_0', prob5_0)
np.save(maxEntDir + '/maxEnt3_prob6', prob6)
np.save(maxEntDir + '/maxEnt3_prob6_0', prob6_0)
np.save(maxEntDir + '/maxEnt3_prob7', prob7)
np.save(maxEntDir + '/maxEnt3_prob7_0', prob7_0)
np.save(maxEntDir + '/maxEnt3_prob8', prob8)
np.save(maxEntDir + '/maxEnt3_prob8_0', prob8_0)

