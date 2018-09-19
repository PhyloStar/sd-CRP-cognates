import random,igraph
from collections import defaultdict
import numpy as np
import itertools as it
from sklearn import metrics
from lingpy import *
rc(schema="asjp")

def sigmoid(score):
    return 1.0/(1.0+np.exp(-score))

def nw(seq_a, seq_b, scores={}, gop=-2.5, gep=-1.75):
    """
    Align two sequences using a flavour of the Needleman-Wunsch algorithm with
    fixed gap opening and gap extension penalties, attributed to Gotoh (1994).

    The scores arg should be a (char_a, char_b): score dict; if a char pair is
    missing, 1/-1 are used as match/mismatch scores.

    Return the best alignment score and one optimal alignment.
    """
    matrix = {}  # (x, y): (score, back)

    for y in range(len(seq_b) + 1):
        for x in range(len(seq_a) + 1):
            cands = []  # [(score, back), ..]

            if x > 0:
                score = matrix[(x-1, y)][0] \
                    + (gep if matrix[(x-1, y)][1] == '←' else gop)
                cands.append((score, '←'))

            if y > 0:
                score = matrix[(x, y-1)][0] \
                    + (gep if matrix[(x, y-1)][1] == '↑' else gop)
                cands.append((score, '↑'))

            if x > 0 and y > 0:
                if (seq_a[x-1], seq_b[y-1]) in scores:
                    score = scores[(seq_a[x-1], seq_b[y-1])]
                else:
                    score = 1 if seq_a[x-1] == seq_b[y-1] else -1
                score += matrix[(x-1, y-1)][0]
                cands.append((score, '.'))
            elif x == 0 and y == 0:
                cands.append((0.0, '.'))

            matrix[(x, y)] = max(cands)

    alignment = []

    while (x, y) != (0, 0):
        if matrix[(x, y)][1] == '←':
            alignment.append((seq_a[x-1], '-'))
            x -= 1
        elif matrix[(x, y)][1] == '↑':
            alignment.append(('-', seq_b[y-1]))
            y -= 1
        else:
            alignment.append((seq_a[x-1], seq_b[y-1]))
            x, y = x-1, y-1

    return matrix[(len(seq_a), len(seq_b))][0], tuple(reversed(alignment))

def readCSV(fname):
    sounds = [x.strip() for x in open('pmi_model/sounds41.txt').readlines()]
    d = defaultdict(lambda: defaultdict())
    cogd = defaultdict(lambda: defaultdict())
    f = open(fname, "r")
    header = f.readline().lower().strip().split("\t")
    asjp_idx = header.index("asjp")
    cog_idx = header.index("cogid")
    print("Reading indexes ", asjp_idx, cog_idx)
    lang_list = defaultdict(int)
    ID = 1
    for line in f:
        line = line.replace("\n","")
        if line.startswith("#") or line.startswith("ID"):
            continue
        else:
            arr = line.split("\t")
            lang = arr[0]
            gloss = arr[2]
            cognID = arr[cog_idx]#.replace("?","")
            if "," in cognID: continue
            if "?" in cognID: continue
            item = arr[asjp_idx]
            #cognID = cognID.split(":")[-1]
            item = item.replace("~","")
            item = item.replace(" ","")
            item = item.replace("%","")
            item = item.replace("*","")
            item = item.replace("$","")
            item = item.replace("\"","")
            item = item.replace("K","")
            item = item.replace("D","d")
            for i in item:
                if i not in sounds:
                    print("Sound %s not in ASJP alphabet" % (i))
            if len(item) < 1: continue
            #print(item)
            d[gloss][ID] = item
            cogd[gloss][ID] = cognID
            lang_list[lang] += 1
            #if lang not in lang_list: lang_list.append(lang)
        ID += 1
    print("{0} languages in the dataset".format(len(lang_list)))
    return d, cogd

def b_cubed(true_labels, labels):
    d = defaultdict()
    precision = [0.0]*len(true_labels)
    recall = [0.0]*len(true_labels)
    
    for t, l in zip(true_labels, labels):
        d[str(l)] = t

    for i, l in enumerate(labels):
        match = 0.0
        prec_denom = 0.0
        recall_denom = 0.0
        for j, m in enumerate(labels):
            if l == m:
                prec_denom += 1.0                
                if true_labels[i] == true_labels[j]:
                    match += 1.0
                    recall_denom += 1.0
            elif l != m:                
                if true_labels[i] == true_labels[j]:
                    recall_denom += 1.0
        precision[i] = match/prec_denom
        recall[i] = match/recall_denom
    #print precision, recall
    avg_precision = np.average(precision)
    avg_recall = np.average(recall)
    avg_f_score = 2.0*avg_precision*avg_recall/(avg_precision+avg_recall)
    return (str(avg_precision), str(avg_recall), str(avg_f_score))

def upgma(pair_dist, gold_dict, gloss):
    import operator
    items_list = list(pair_dist.keys())
    #print(items_list)
    #print("Gloss ",gloss)
    dist_mat = defaultdict(float)
    for x, y in it.permutations(list(range(len(items_list))), r=2):
        dist_mat[(x,),(y,)] = pair_dist[items_list[x]][items_list[y]]
    
    #clusters = list(zip(*list(dist_mat.keys()))[0]
    clusters = set([x[0] for x in dist_mat.keys()])
    #print("Clusters in the beginnining", clusters)
    #print(max(dist_mat.values()))
    #print(max(dist_mat))
    #if len(items_list) == 1: return
    while(1):
        if max(dist_mat.values()) > 0:
            clust_i, clust_j = max(dist_mat, key=dist_mat.get)
            #print("To be merged ", clust_i, clust_j, dist_mat[clust_i, clust_j])

            new_clust = clust_i + clust_j
            
            #print("Before Deletion ", set([x[0] for x in dist_mat.keys()]))
            temp_clusts = list(dist_mat.keys())

            clusters = set([x[0] for x in temp_clusts])
            for t in clusters:
                if clust_i == t or clust_j == t: continue
                denom = len(new_clust)*len(list(t))
                for x, y in it.product(new_clust, list(t)):
                    dist_mat[new_clust,t] += pair_dist[items_list[x]][items_list[y]]
                dist_mat[new_clust,t] = dist_mat[new_clust,t]/denom
                dist_mat[t,new_clust] = dist_mat[new_clust,t]

            for clust_a, clust_b in temp_clusts:
                if clust_a in [clust_i, clust_j] or clust_b in [clust_i, clust_j]:
                    #print("Deleted ", clust_a, clust_b)
                    del dist_mat[clust_a, clust_b]
            
            if len(clusters) == 2: 
                dist_mat[new_clust,new_clust] = 1.0
                break
        else:
            #print("exiting loop")
            break            
    #print("Final ",dist_mat.keys())
    clusters = [[items_list[y] for y in z] for z in set([x[0] for x in dist_mat.keys()])]
    #print(clusters)
    #print(set([x[0] for x in dist_mat.keys()]))
    #pred_clusters = [  for z_idx, z in enumerate(clusters)]
    predicted_labels, gold_labels = [], []
    for k_idx, k in enumerate(clusters):
        for k_item in k:
            predicted_labels.append(int(k_idx))
            gold_labels.append(gold_dict[int(k_item.split("::")[1])])
            
    p, r, f_score = b_cubed(gold_labels,predicted_labels)
    ari = metrics.adjusted_rand_score(gold_labels, predicted_labels)
    #print(clusters)
    #print(gold_labels, str(len(set(gold_labels))), predicted_labels)
    assert(len(set(gold_labels)) == len(set(gold_dict.values())))
    #print(gloss, f_score, ari, str(len(clusters)), str(len(set(gold_labels))))
    scores = [str(p), str(r), str(f_score), str(ari), str(len(clusters)), str(len(set(gold_labels)))]
    return scores



def igraph_clustering(matrix, lex_list, threshold=0.57, method='labelprop'):
    """
    Method computes Infomap clusters from pairwise distance data.
    """
    random.seed(1234)
    G = igraph.Graph()
    vertex_weights = []
    for i in range(len(matrix)):
        G.add_vertex(i)
        vertex_weights += [0]
    
    # variable stores edge weights, if they are not there, the network is
    # already separated by the threshold
    weights = None
    for i,row in enumerate(matrix):
        for j,cell in enumerate(row):
            if i < j:
                if cell <= threshold:
                    G.add_edge(i, j, weight=1-cell, distance=cell)
                    weights = 'weight'

    if method == 'infomap':
        comps = G.community_infomap(edge_weights=weights,
                vertex_weights=None)
        
    elif method == 'labelprop':
        comps = G.community_label_propagation(weights=weights,
                initial=None, fixed=None)

    elif method == 'ebet':
        dg = G.community_edge_betweenness(weights=weights)
        oc = dg.optimal_count
        comps = False
        while oc <= len(G.vs):
            try:
                comps = dg.as_clustering(dg.optimal_count)
                break
            except:
                oc += 1
        if not comps:
            print('Failed...')
            comps = list(range(len(G.sv)))
            input()
    elif method == 'multilevel':
        comps = G.community_multilevel(return_levels=False)
    elif method == 'spinglass':
        comps = G.community_spinglass()

    D = {}
    for i,comp in enumerate(comps.subgraphs()):
        vertices = [v['name'] for v in comp.vs]
        for vertex in vertices:
            D[vertex] = i+1


    return D


def ldn(a, b):
    """
    Leventsthein distance normalized
    :param a: word
    :type a: str
    :param b: word
    :type b: str
    :return: distance score
    :rtype: float
    """
    m = [];
    la = len(a) + 1;
    lb = len(b) + 1
    for i in range(0, la):
        m.append([])
        for j in range(0, lb): m[i].append(0)
        m[i][0] = i
    for i in range(0, lb): m[0][i] = i
    for i in range(1, la):
        for j in range(1, lb):
            s = m[i - 1][j - 1]
            if (a[i - 1] != b[j - 1]): s = s + 1
            m[i][j] = min(m[i][j - 1] + 1, m[i - 1][j] + 1, s)
    la = la - 1;
    lb = lb - 1
    return float(m[la][lb]) / float(max(la, lb))

def prefix(a,b):
    count = 0.0
    for x, y in zip(a,b):
        if x==y:
            count+=1.0
        else:
            break
    return count

def bigrams(a, b):
    count = 0.0
    for x, y in zip(a[:-1], b[1:]):
        if x==y: count+=1.0
    return count


def read_pmidict(pmi_fname):
    scores = {}
    for line in open(pmi_fname, "r"):
        x, y, s = line.replace("\n","").split("\t")
        scores[x,y]=float(s)
    return scores

