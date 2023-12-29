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
    parser.add_argument("-c", "--canonical_ss", required=True)
    parser.add_argument("-a", "--all_ss", required=True)
    parser.add_argument("-g", "--genome", required=True)
    parser.add_argument("-m", "--maxent_dir", required=True)
    
    parser.add_argument("--learn_sres", action="store_true")
    parser.add_argument("--max_learned_scores", type=int, default = 1000)
    
    parser.add_argument("--learning_seed", choices = ['none', 'real-decoy'], default = 'none')
    parser.add_argument("--max_learned_weights", type=int, default = 15)
    
    parser.add_argument("--print_predictions", action="store_true")
    
    opts = parser.parse_args()
    return opts

args = parse_arguments()

# Load data from arguments
maxEntDir = args.maxent_dir
genome = SeqIO.to_dict(SeqIO.parse(args.genome, "fasta"))
canonical = pd.read_csv(args.canonical_ss, sep = '\t', engine='python', index_col=0, header = None)
canonical.index = canonical.index.map(str)
allSS = pd.read_csv(args.all_ss, sep = '\t', engine='python', index_col=0, header = None) 
allSS.index = allSS.index.map(str)

genes = {}
for gene in canonical.index:
    txstart = canonical.loc[gene,4]-1
    txend = canonical.loc[gene,5]
    chrom = canonical.loc[gene,2][3:]
    if not np.array_equiv(['a', 'c', 'g', 't'],
         np.unique(genome[chrom][canonical.loc[gene,4]-1:canonical.loc[gene,5]].seq.lower())): 
#         print(gene, "nonstandard base")
        continue
    
    name = gene
    geneID = gene
    description = gene + ' GeneID:'+gene + ' TranscriptID:Canonical' + ' Chromosome:' + canonical.loc[gene,2] + \
    ' Start:'+str(txstart) + ' Stop:'+str(txend) + ' Strand:'+canonical.loc[gene,3]

    if canonical.loc[gene,3] == '-':
        seq = genome[chrom][canonical.loc[gene,4]-1:canonical.loc[gene,5]].seq.reverse_complement()
    elif canonical.loc[gene,3] == '+': 
        seq = genome[chrom][canonical.loc[gene,4]-1:canonical.loc[gene,5]].seq
    else:
        print(gene, "strand error")
    
    genes[gene] = SeqRecord(seq, name = name, id = geneID, description = description)

# Additional parameters
sreEffect = 80
sreEffect3_intron = sreEffect + 19
sreEffect3_exon = sreEffect + 3
sreEffect5_intron = sreEffect + 5
sreEffect5_exon = sreEffect + 3
np.seterr(divide='ignore')

kmer = 6
E = 0
I = 1
B5 = 5
B3 = 3
train_size = 4000
test_size = 1000
score_learning_rate = .01


# Get training, validation, generalization, and test sets
testGenes = canonical[(canonical[1] == 0)&canonical[2].isin(['chr2', 'chr4'])].index 
testGenes = np.intersect1d(testGenes, list(genes.keys()))

trainGenes = canonical.index
trainGenes = np.intersect1d(trainGenes, list(genes.keys()))
trainGenes = np.setdiff1d(trainGenes, testGenes)

lengthsOfGenes = np.array([len(str(genes[gene].seq)) for gene in trainGenes])
trainGenes = trainGenes[lengthsOfGenes > sreEffect3_intron]
lengthsOfGenes = np.array([len(str(genes[gene].seq)) for gene in trainGenes])
validationGenes = trainGenes[lengthsOfGenes < 200000]

generalizationGenes = np.intersect1d(validationGenes, canonical[canonical[1] == 0].index)
if len(generalizationGenes) > test_size: generalizationGenes = np.random.choice(generalizationGenes, test_size, replace = False)
validationGenes = np.setdiff1d(validationGenes, generalizationGenes)
if len(validationGenes) > test_size: validationGenes = np.random.choice(validationGenes, test_size, replace = False)

trainGenes = np.setdiff1d(trainGenes, generalizationGenes)
trainGenes = np.setdiff1d(trainGenes, validationGenes)
if len(trainGenes) > train_size: trainGenes = np.random.choice(trainGenes, train_size, replace = False)

cannonical_annotations = {}
annotations = {}
for gene in genes.keys():
    annnotation = []
    info = genes[gene].description.split(' ')
    
    if canonical.loc[gene,6] == ',': 
        cannonical_annotations[gene] = [(0, int(info[5][5:]) - int(info[4][6:]) - 1)]
        continue
        
    exonEnds = [int(start)-1 for start in canonical.loc[gene,6].split(',')[:-1]] + [int(info[5][5:])] # intron starts -> exon ends
    exonStarts = [int(info[4][6:])] + [int(end)+1 for end in canonical.loc[gene,7].split(',')[:-1]] # exon starts -> intron ends
    exonEnds[-1] -= 1
    exonStarts[0] += 2
    
    if info[6] == 'Strand:-': 
        stop = int(info[5][5:])
        for i in range(len(exonEnds),0,-1):
            annnotation.append((stop - exonEnds[i-1] - 1, stop - exonStarts[i-1] + 1))        
        
    elif info[6] == 'Strand:+': 
        start = int(info[4][6:])
        for i in range(len(exonEnds)):
            annnotation.append((exonStarts[i] - start - 2, exonEnds[i] - start))
            
    cannonical_annotations[gene] = annnotation
    annotations[gene] = {'Canonical' : annnotation}
trueSeqs = trueSequencesCannonical(genes, cannonical_annotations, E, I, B3, B5) 

# Learning seed
sreScores_exon = np.ones(4**kmer)
sreScores3_exon = np.ones(4**kmer)
sreScores5_exon = np.ones(4**kmer)

sreScores_intron = np.ones(4**kmer)
sreScores3_intron = np.ones(4**kmer)
sreScores5_intron = np.ones(4**kmer)
        
lengths = np.array([len(str(genes[gene].seq)) for gene in validationGenes])
sequences = [str(genes[gene].seq) for gene in validationGenes] 

exonicSREs5s = np.zeros((len(lengths), max(lengths)-kmer+1))
exonicSREs3s = np.zeros((len(lengths), max(lengths)-kmer+1))
intronicSREs5s = np.zeros((len(lengths), max(lengths)-kmer+1))
intronicSREs3s = np.zeros((len(lengths), max(lengths)-kmer+1))

scored_sequences_5, scored_sequences_3  = score_sequences(sequences = sequences, exonicSREs5s = exonicSREs5s, exonicSREs3s = exonicSREs3s, intronicSREs5s = intronicSREs5s, intronicSREs3s = intronicSREs3s, k = kmer, sreEffect5_exon = sreEffect5_exon, sreEffect5_intron = sreEffect5_intron, sreEffect3_exon = sreEffect3_exon, sreEffect3_intron = sreEffect3_intron, meDir = maxEntDir)
ssSens, ssPrec, f1, score_cutoff = cass_accuracy_metrics(scored_sequences_5, scored_sequences_3, validationGenes, trueSeqs)

if args.learning_seed == 'real-decoy' and args.learn_sres:
    me5 = maxEnt5(trainGenes, genes, maxEntDir)
    me3 = maxEnt3(trainGenes, genes, maxEntDir)
    
    tolerance = .5
    decoySS = {}
    for gene in trainGenes:
        decoySS[gene] = np.zeros(len(genes[gene]), dtype = int)
        
    # 5'SS
    five_scores = []
    for gene in trainGenes:
        for score in np.log2(me5[gene][trueSeqs[gene] == B5][1:]): five_scores.append(score)
    five_scores = np.array(five_scores)

    five_scores_tracker = np.flip(np.sort(list(five_scores)))

    for score in five_scores_tracker:
        np.random.shuffle(trainGenes)
        g = 0
        while g < len(trainGenes):
            gene = trainGenes[g]
            g += 1
            true_ss = get_all_5ss(gene, allSS, genes)
            used_sites = np.nonzero(decoySS[gene] == B5)[0]

            gene5s = np.log2(me5[gene])
            sort_inds = np.argsort(gene5s)
            sort_inds = sort_inds[~np.in1d(sort_inds, true_ss)]
            sort_inds = sort_inds[~np.in1d(sort_inds, used_sites)]
            L = len(sort_inds)
            gene5s = gene5s[sort_inds]

            up_i = np.searchsorted(gene5s, score, 'left')
            down_i = up_i - 1
            if down_i >= L: down_i = L-1
            if up_i >= L: up_i = L-1

            if abs(score - gene5s[down_i]) < tolerance and decoySS[gene][sort_inds[down_i]] == 0:
                decoySS[gene][sort_inds[down_i]] = B5
                g = len(trainGenes)

            elif abs(score - gene5s[up_i]) < tolerance and decoySS[gene][sort_inds[up_i]] == 0:
                decoySS[gene][sort_inds[up_i]] = B5
                g = len(trainGenes)
                
    # 3'SS 
    three_scores = []
    for gene in trainGenes:
        for score in np.log2(me3[gene][trueSeqs[gene] == B3][:-1]): three_scores.append(score)
    three_scores = np.array(three_scores)

    three_scores_tracker = np.flip(np.sort(list(three_scores)))

    for score in three_scores_tracker:
        np.random.shuffle(trainGenes)
        g = 0
        while g < len(trainGenes):
            gene = trainGenes[g]
            g += 1
            true_ss = get_all_3ss(gene, allSS, genes)
            used_sites = np.nonzero(decoySS[gene] == B3)[0]

            gene3s = np.log2(me3[gene])
            sort_inds = np.argsort(gene3s)
            sort_inds = sort_inds[~np.in1d(sort_inds, true_ss)]
            sort_inds = sort_inds[~np.in1d(sort_inds, used_sites)]
            L = len(sort_inds)
            gene3s = gene3s[sort_inds]

            up_i = np.searchsorted(gene3s, score, 'left')
            down_i = up_i - 1
            if down_i >= L: down_i = L-1
            if up_i >= L: up_i = L-1

            if abs(score - gene3s[down_i]) < tolerance and decoySS[gene][sort_inds[down_i]] == 0:
                decoySS[gene][sort_inds[down_i]] = B3
                g = len(trainGenes)

            elif abs(score - gene3s[up_i]) < tolerance and decoySS[gene][sort_inds[up_i]] == 0:
                decoySS[gene][sort_inds[up_i]] = B3
                g = len(trainGenes)

    (sreScores_intron, sreScores_exon, sreScores3_intron, sreScores3_exon, sreScores5_intron, sreScores5_exon) = get_hexamer_real_decoy_scores(trainGenes, trueSeqs, decoySS, genes, kmer = kmer, sreEffect5_exon = sreEffect5_exon, sreEffect5_intron = sreEffect5_intron, sreEffect3_exon = sreEffect3_exon, sreEffect3_intron = sreEffect3_intron)
    sreScores3_intron = sreScores_intron
    sreScores3_exon = sreScores_exon
    sreScores5_intron = np.copy(sreScores_intron)
    sreScores5_exon = np.copy(sreScores_exon)
    
    # Learn weight
    lengths = np.array([len(str(genes[gene].seq)) for gene in validationGenes])
    sequences = [str(genes[gene].seq) for gene in validationGenes]    
    
    step_size = 1
    sre_weights = [0, step_size]
    scores = []
    cutoffs = []
    for sre_weight in sre_weights:
        exonicSREs5s = np.zeros((len(lengths), max(lengths)-kmer+1))
        exonicSREs3s = np.zeros((len(lengths), max(lengths)-kmer+1))
        intronicSREs5s = np.zeros((len(lengths), max(lengths)-kmer+1))
        intronicSREs3s = np.zeros((len(lengths), max(lengths)-kmer+1))
            
        for g, sequence in enumerate(sequences):
            exonicSREs5s[g,:lengths[g]-kmer+1] = np.log(np.array(sreScores_single(sequence.lower(), np.exp(np.log(sreScores5_exon)*sre_weight), kmer)))
            exonicSREs3s[g,:lengths[g]-kmer+1] = np.log(np.array(sreScores_single(sequence.lower(), np.exp(np.log(sreScores3_exon)*sre_weight), kmer)))
            intronicSREs5s[g,:lengths[g]-kmer+1] = np.log(np.array(sreScores_single(sequence.lower(), np.exp(np.log(sreScores5_intron)*sre_weight), kmer)))
            intronicSREs3s[g,:lengths[g]-kmer+1] = np.log(np.array(sreScores_single(sequence.lower(), np.exp(np.log(sreScores3_intron)*sre_weight), kmer)))
                   
        scored_sequences_5, scored_sequences_3  = score_sequences(sequences = sequences, exonicSREs5s = exonicSREs5s, exonicSREs3s = exonicSREs3s, intronicSREs5s = intronicSREs5s, intronicSREs3s = intronicSREs3s, k = kmer, sreEffect5_exon = sreEffect5_exon, sreEffect5_intron = sreEffect5_intron, sreEffect3_exon = sreEffect3_exon, sreEffect3_intron = sreEffect3_intron, meDir = maxEntDir)
        ssSens, ssPrec, f1, score_cutoff = cass_accuracy_metrics(scored_sequences_5, scored_sequences_3, validationGenes, trueSeqs)
        scores.append(f1)
     
    scores = np.array(scores)
    sre_weights = np.array(sre_weights)
    cutoffs = np.array(cutoffs)
    
    while len(scores) < 15:
        i = np.argmax(scores)
        if i == len(scores) - 1:
            sre_weights_test = [sre_weights[-1] + step_size]
        elif i == 0:
            sre_weights_test = [sre_weights[1]/2]
        else:
            sre_weights_test = [sre_weights[i]/2 + sre_weights[i-1]/2, sre_weights[i]/2 + sre_weights[i+1]/2]
            
        for sre_weight in sre_weights_test: 
            sre_weights = np.append(sre_weights, sre_weight)
            exonicSREs5s = np.zeros((len(lengths), max(lengths)-kmer+1))
            exonicSREs3s = np.zeros((len(lengths), max(lengths)-kmer+1))
            intronicSREs5s = np.zeros((len(lengths), max(lengths)-kmer+1))
            intronicSREs3s = np.zeros((len(lengths), max(lengths)-kmer+1))
    
            for g, sequence in enumerate(sequences):
                exonicSREs5s[g,:lengths[g]-kmer+1] = np.log(np.array(sreScores_single(sequence.lower(), np.exp(np.log(sreScores5_exon)*sre_weight), kmer)))
                exonicSREs3s[g,:lengths[g]-kmer+1] = np.log(np.array(sreScores_single(sequence.lower(), np.exp(np.log(sreScores3_exon)*sre_weight), kmer)))
                intronicSREs5s[g,:lengths[g]-kmer+1] = np.log(np.array(sreScores_single(sequence.lower(), np.exp(np.log(sreScores5_intron)*sre_weight), kmer)))
                intronicSREs3s[g,:lengths[g]-kmer+1] = np.log(np.array(sreScores_single(sequence.lower(), np.exp(np.log(sreScores3_intron)*sre_weight), kmer)))
                    
            scored_sequences_5, scored_sequences_3  = score_sequences(sequences = sequences, exonicSREs5s = exonicSREs5s, exonicSREs3s = exonicSREs3s, intronicSREs5s = intronicSREs5s, intronicSREs3s = intronicSREs3s, k = kmer, sreEffect5_exon = sreEffect5_exon, sreEffect5_intron = sreEffect5_intron, sreEffect3_exon = sreEffect3_exon, sreEffect3_intron = sreEffect3_intron, meDir = maxEntDir)
            ssSens, ssPrec, f1, score_cutoff = cass_accuracy_metrics(scored_sequences_5, scored_sequences_3, validationGenes, trueSeqs)
            
            scores = np.append(scores, f1)
            cutoffs = np.append(cutoffs, score_cutoff)

        scores = scores[np.argsort(sre_weights)]
        sre_weights = sre_weights[np.argsort(sre_weights)]
        
    # Set up scores for score learning
    sre_weight = sre_weights[np.argmax(scores)]
    
    sreScores_exon = np.exp(np.log(sreScores5_exon)*sre_weight)
    sreScores5_exon = np.exp(np.log(sreScores5_exon)*sre_weight)
    sreScores3_exon = np.exp(np.log(sreScores3_exon)*sre_weight)
    sreScores_intron = np.exp(np.log(sreScores5_intron)*sre_weight)
    sreScores5_intron = np.exp(np.log(sreScores5_intron)*sre_weight)
    sreScores3_intron = np.exp(np.log(sreScores3_intron)*sre_weight)
      
# Learning  
if args.learn_sres:
    lengthsOfGenes = np.array([len(str(genes[gene].seq)) for gene in trainGenes])
    trainGenesShort = trainGenes[lengthsOfGenes < 200000]
    np.random.shuffle(trainGenesShort)
    trainGenesShort = np.array_split(trainGenesShort, 4)
    
    trainGenes1 = trainGenesShort[0]
    trainGenes2 = trainGenesShort[1]
    trainGenes3 = trainGenesShort[2]
    trainGenes4 = trainGenesShort[3]
    trainGenesTest = generalizationGenes
    held_f1 = -1
    
    learning_counter = -1
    doneTime = 10
    while 1 < doneTime:
        learning_counter += 1
        if learning_counter > args.max_learned_scores: break 
        update_scores = True
        
        if learning_counter%5 == 1: trainGenesSub = np.copy(trainGenes1)
        elif learning_counter%5 == 2: trainGenesSub = np.copy(trainGenes2)
        elif learning_counter%5 == 3: trainGenesSub = np.copy(trainGenes3)
        elif learning_counter%5 == 4: trainGenesSub = np.copy(trainGenes4)
        else: 
            trainGenesSub = np.copy(trainGenesTest)
            update_scores = False
        
        lengths = np.array([len(str(genes[gene].seq)) for gene in trainGenesSub])
        sequences = [str(genes[gene].seq) for gene in trainGenesSub]
        
        exonicSREs5s = np.zeros((len(lengths), max(lengths)-kmer+1))
        exonicSREs3s = np.zeros((len(lengths), max(lengths)-kmer+1))
        intronicSREs5s = np.zeros((len(lengths), max(lengths)-kmer+1))
        intronicSREs3s = np.zeros((len(lengths), max(lengths)-kmer+1))

        for g, sequence in enumerate(sequences):
            exonicSREs5s[g,:lengths[g]-kmer+1] = np.log(np.array(sreScores_single(sequence.lower(), sreScores5_exon, kmer)))
            exonicSREs3s[g,:lengths[g]-kmer+1] = np.log(np.array(sreScores_single(sequence.lower(), sreScores3_exon, kmer)))
            intronicSREs5s[g,:lengths[g]-kmer+1] = np.log(np.array(sreScores_single(sequence.lower(), sreScores5_intron, kmer)))
            intronicSREs3s[g,:lengths[g]-kmer+1] = np.log(np.array(sreScores_single(sequence.lower(), sreScores3_intron, kmer)))

        scored_sequences_5, scored_sequences_3  = score_sequences(sequences = sequences, exonicSREs5s = exonicSREs5s, exonicSREs3s = exonicSREs3s, intronicSREs5s = intronicSREs5s, intronicSREs3s = intronicSREs3s, k = kmer, sreEffect5_exon = sreEffect5_exon, sreEffect5_intron = sreEffect5_intron, sreEffect3_exon = sreEffect3_exon, sreEffect3_intron = sreEffect3_intron, meDir = maxEntDir)
        
        if update_scores: ssSens, ssPrec, f1, score_cutoff = cass_accuracy_metrics(scored_sequences_5, scored_sequences_3, trainGenesSub, trueSeqs)
        else: ssSens, ssPrec, f1 = cass_accuracy_metrics_set_cutoff(scored_sequences_5, scored_sequences_3, trainGenesSub, trueSeqs, score_cutoff)
        
        # Get the False Negatives and False Positives
        falsePositives = {}
        falseNegatives = {}
        for j, gene in enumerate(trainGenesSub):
            falsePositives[gene] = np.zeros(len(genes[gene]), dtype = int)
            falseNegatives[gene] = np.zeros(len(genes[gene]), dtype = int)
            
            predThrees = np.nonzero(np.log2(scored_sequences_3[j,:]) >= score_cutoff)[0]
            trueThrees = np.nonzero(trueSeqs[gene] == B3)[0]
            gtex_3_ss = get_all_3ss(gene, allSS, genes)

            predFives = np.nonzero(np.log2(scored_sequences_5[j,:]) >= score_cutoff)[0]
            trueFives = np.nonzero(trueSeqs[gene] == B5)[0]
            gtex_5_ss = get_all_5ss(gene, allSS, genes)

            falsePositives[gene][np.setdiff1d(predThrees, trueThrees)] = B3
            falsePositives[gene][np.setdiff1d(predFives, trueFives)] = B5

            falseNegatives[gene][np.setdiff1d(trueThrees, predThrees)] = B3
            falseNegatives[gene][np.setdiff1d(trueFives, predFives)] = B5
        
        if update_scores:
            set1_counts_5_intron, set1_counts_5_exon, set1_counts_3_intron, set1_counts_3_exon, set2_counts_5_intron, set2_counts_5_exon, set2_counts_3_intron, set2_counts_3_exon = get_hexamer_counts(trainGenesSub, falseNegatives, falsePositives, genes, kmer = kmer, sreEffect5_exon = sreEffect5_exon, sreEffect5_intron = sreEffect5_intron, sreEffect3_exon = sreEffect3_exon, sreEffect3_intron = sreEffect3_intron)
        
            set1_counts_intron = set1_counts_5_intron + set1_counts_3_intron
            set1_counts_exon = set1_counts_5_exon + set1_counts_3_exon
            set2_counts_intron = set2_counts_5_intron + set2_counts_3_intron
            set2_counts_exon = set2_counts_5_exon + set2_counts_3_exon
        
            psuedocount_denominator_intron = np.sum(set1_counts_intron) + np.sum(set2_counts_intron)
            set1_counts_intron = set1_counts_intron + np.sum(set1_counts_intron) / psuedocount_denominator_intron
            set2_counts_intron = set2_counts_intron + np.sum(set2_counts_intron) / psuedocount_denominator_intron
    
            frequency_ratio_intron = set1_counts_intron/np.sum(set1_counts_intron) / (set2_counts_intron/np.sum(set2_counts_intron))
            sreScores_intron *= frequency_ratio_intron**score_learning_rate 
        
            psuedocount_denominator_exon = np.sum(set1_counts_exon) + np.sum(set2_counts_exon)
            set1_counts_exon = set1_counts_exon + np.sum(set1_counts_exon) / psuedocount_denominator_exon
            set2_counts_exon = set2_counts_exon + np.sum(set2_counts_exon) / psuedocount_denominator_exon
    
            frequency_ratio_exon = set1_counts_exon/np.sum(set1_counts_exon) / (set2_counts_exon/np.sum(set2_counts_exon))
            sreScores_exon *= frequency_ratio_exon**score_learning_rate 
        
            sreScores3_intron = sreScores_intron
            sreScores3_exon = sreScores_exon
            sreScores5_intron = sreScores_intron
            sreScores5_exon = sreScores_exon
                
        else:
            if f1 >= held_f1:  # Hold onto the scores with the highest f1 performance
                held_f1 = np.copy(f1)
                held_sreScores5_exon = np.copy(sreScores5_exon)
                held_sreScores3_exon = np.copy(sreScores3_exon)
                held_sreScores5_intron = np.copy(sreScores5_intron)
                held_sreScores3_intron = np.copy(sreScores3_intron)
                held_score_cutoff = np.copy(score_cutoff)
            else: doneTime = 0 # Stop learning if the performance has decreased
         
    sreScores5_exon = np.copy(held_sreScores5_exon)
    sreScores3_exon = np.copy(held_sreScores3_exon)
    sreScores5_intron = np.copy(held_sreScores5_intron)
    sreScores3_intron = np.copy(held_sreScores3_intron)
    score_cutoff = np.copy(held_score_cutoff)

# Filter test set
lengthsOfGenes = np.array([len(str(genes[gene].seq)) for gene in testGenes])
testGenes = testGenes[lengthsOfGenes > sreEffect3_intron]

notShortIntrons = []
for gene in testGenes:
    trueThrees = np.nonzero(trueSeqs[gene] == B3)[0]
    trueFives = np.nonzero(trueSeqs[gene] == B5)[0]
    
    n_fives = np.sum(trueSeqs[gene] == B5)
    n_threes = np.sum(trueSeqs[gene] == B3)
    
    if n_fives != n_threes: notShortIntrons.append(False)
    elif np.min(trueThrees-trueFives+1) < 25: notShortIntrons.append(False)
    else: notShortIntrons.append(True)
    
notShortIntrons = np.array(notShortIntrons)
testGenes = testGenes[notShortIntrons]
lengths = np.array([len(str(genes[gene].seq)) for gene in testGenes])
sequences = [str(genes[gene].seq) for gene in testGenes]

exonicSREs5s = np.zeros((len(lengths), max(lengths)-kmer+1))
exonicSREs3s = np.zeros((len(lengths), max(lengths)-kmer+1))
intronicSREs5s = np.zeros((len(lengths), max(lengths)-kmer+1))
intronicSREs3s = np.zeros((len(lengths), max(lengths)-kmer+1))

for g, sequence in enumerate(sequences):
    exonicSREs5s[g,:lengths[g]-kmer+1] = np.log(np.array(sreScores_single(sequence.lower(), sreScores5_exon, kmer)))
    exonicSREs3s[g,:lengths[g]-kmer+1] = np.log(np.array(sreScores_single(sequence.lower(), sreScores3_exon, kmer)))
    intronicSREs5s[g,:lengths[g]-kmer+1] = np.log(np.array(sreScores_single(sequence.lower(), sreScores5_intron, kmer)))
    intronicSREs3s[g,:lengths[g]-kmer+1] = np.log(np.array(sreScores_single(sequence.lower(), sreScores3_intron, kmer)))

scored_sequences_5, scored_sequences_3  = score_sequences(sequences = sequences, exonicSREs5s = exonicSREs5s, exonicSREs3s = exonicSREs3s, intronicSREs5s = intronicSREs5s, intronicSREs3s = intronicSREs3s, k = kmer, sreEffect5_exon = sreEffect5_exon, sreEffect5_intron = sreEffect5_intron, sreEffect3_exon = sreEffect3_exon, sreEffect3_intron = sreEffect3_intron, meDir = maxEntDir)
ssSens, ssPrec, f1 = cass_accuracy_metrics_set_cutoff(scored_sequences_5, scored_sequences_3, testGenes, trueSeqs, score_cutoff)

if args.print_predictions: 
    for g, gene in enumerate(testGenes):
        L = lengths[g]
        predThrees = np.nonzero(np.log2(scored_sequences_3[g,:L]) > score_cutoff)[0] 
        predFives = np.nonzero(np.log2(scored_sequences_5[g,:L]) > score_cutoff)[0] 
        trueThrees = np.nonzero(trueSeqs[gene] == B3)[0]
        trueFives = np.nonzero(trueSeqs[gene] == B5)[0]
    
        print(gene)
        print("\tAnnotated Fives:", trueFives, "Predicted Fives:", predFives)
        print("\tAnnotated Threes:", trueThrees, "Predicted Threes:", predThrees)

print("Final Test Metrics", "Recall", '{:.2f}'.format(ssSens*100), "Precision", '{:.2f}'.format(ssPrec*100), "f1", '{:.2f}'.format(f1*100)) 


