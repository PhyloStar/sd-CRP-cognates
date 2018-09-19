'''
Created on Mar 7, 2016

@author: taraka
'''
from collections import defaultdict
from itertools import combinations, combinations_with_replacement, product
import numpy as np
import random, pprint, utils, math

from scipy import stats
from sklearn import metrics
np.random.seed(1334)
random.seed(1334)
import glob, sys, igraph
from lingpy import *
import os.path

import argparse
parser = argparse.ArgumentParser(description="CRP for word lists")
parser.add_argument("-ca","--calpha", type= float, help="CRP alpha", default=0.1)
parser.add_argument("-t", "--thd", type= float, help="A number between 0 and 1 for clustering", default=0.57)
parser.add_argument("-i","--infile", type= str, help="input file name", default="")
parser.add_argument("-o","--outfile", type= str, help="output file name", default="temp")
parser.add_argument("-A","--in_alphabet", type= str, help="input alphabet asjp, sca, dolgo", default="asjp")
parser.add_argument("-clf","--classifier", type= str, help="jaeger PMI, svm, nw, sca", default="svm")
parser.add_argument("-w","--weights", type= str, help="SVM weights file name", default="svm_params_HK")
parser.add_argument("--pmi_flag", help="add PMI", action='store_true')
parser.add_argument("--sca_flag", help="add SCA", action='store_true')
parser.add_argument("--sample", help="sample crp alpha", action='store_true')
args = parser.parse_args()

#parser.add_argument("-c","--clust_algo", type= str, help="clustering algorithm name: infomap, labelprop, upgma, socher, crp", default="infomap")

max_iter = 10

rc(schema=args.in_alphabet)

fh = open(args.outfile, "w")

if args.sample: max_iter = 100

#print("SVM Weights ",  svm_W, svm_b,sep="\n")

print(args.infile)

scaler_alpha = 1.0
exp_lambda = 10.0

gp1 = -2.49302792222
gp2 = -1.70573165621


lodict = defaultdict()
f = open('pmi_model/sounds41.txt')
sounds = np.array([x.strip() for x in f.readlines()])
f.close()

f = open('pmi_model/pmi-world.txt','r')
l = f.readlines()
f.close()
logOdds = np.array([x.strip().split() for x in l],np.float)

for i in range(len(sounds)):#Initiate sound dictionary
    for j in range(len(sounds)):
        lodict[sounds[i],sounds[j]] = logOdds[i,j]

#lodict = utils.read_pmidict("temp.pmi")

if os.path.exists(args.weights):
    if args.sca_flag and  args.pmi_flag:
        args.weights = "svm_params_HK.pmi.sca"
    elif args.pmi_flag:
        args.weights = "svm_params_HK.pmi"
    elif args.sca_flag:
        args.weights = "svm_params_HK.sca"

    print("Reading file name {}".format(args.weights))

    lines = open(args.weights,"r").read()
    W = list(map(float, lines.split(",")))
    print("Number of feature weights {}".format(len(W)))
    svm_W, svm_b  = W[:-1], W[-1]
#else:
#    svm_W, svm_b = svm_HK.svm_train()

#print(svm_W, svm_b)

def HK_test(w1, w2, pmi_flag = False, sca_flag=False):
    #
    fea_vec = [utils.ldn(w1, w2), utils.prefix(w1,w2), utils.bigrams(w1, w2),len(w1), len(w2), abs(len(w1)-len(w2))]
    if pmi_flag:
        fea_vec += [utils.sigmoid(utils.nw(w1, w2, scores=lodict)[0])]
    if sca_flag:
        fea_vec += [my_SCA(w1,w2)]
    #print(fea_vec)  
    return np.array(fea_vec)
    
def my_SCA(w1, w2, distance_flag=True):
    a = Pairwise(w1, w2)
    a.align(distance = distance_flag)
    return a.alignments[0][-1]

def consensus_clustering(d, items_list, gold_dict, threshold=80):
    n_items = len(items_list)
    g = igraph.Graph(n_items, directed=True)
    for v in range(n_items):
        g.vs[v]["name"] = items_list[v]
    
    for i, i1 in enumerate(items_list):
        for j, j1 in enumerate(items_list):
            if d[i1,j1] >= threshold:
                g.add_edges([(i, j)])

    predicted_labels, gold_labels = [], []

    #print(g.clusters(mode="weak"))

    for k_idx, k in enumerate(g.clusters(mode="weak")):
        for x in k:
            predicted_labels.append(k_idx)
            gold_labels.append(gold_dict[int(items_list[x].split("::")[1])])
    ari = metrics.adjusted_rand_score(gold_labels, predicted_labels)
    return list(utils.b_cubed(gold_labels,predicted_labels))+[ari]

def socherCRP(pair_dist, gold_dict, gloss):
    """Socher et al. (2011) sampling based CRP. Does nothing much but moves the items around.
    """
    alpha = float(args.calpha)
    #alpha = random.expovariate(exp_lambda)
    items_list = list(pair_dist.keys())####Modify this
    assert len(items_list) >=1
    n_items = len(items_list)

    cluster_idx = [[x] for x in items_list]
    n_iter = 1
    clusters_vec, cluster_idxs = [], []
    cluster_idxs.append([x for x in cluster_idx])
    bcubed_prec, bcubed_recall, bcubed_fscore, ari_vec = [], [], [], []
    g = igraph.Graph(n_items, directed=True)
    for v in range(n_items):
        g.vs[v]["name"] = items_list[v].split("::")[0]
    
    for n_iter in range(max_iter):
        #random.shuffle(items_list)    

        for i, item in enumerate(items_list):
            #print("Item ",i, item)
            
            sim_vec = [0.0]*n_items
            
            for j, v in enumerate(items_list):
                if i == j:
                    sim_vec[j] = alpha*pair_dist[v][v]
                    continue
                #print("\tTarget ",j, items_list[j])
                if g.are_connected(i,j):
                    g.delete_edges([(i,j)])

                sitbehind = g.subcomponent(j, mode="in")
                sSim = 0.0
                #print(sitbehind)
                for s in sitbehind:
                    sSim += pair_dist[item][items_list[s]]
                sim_vec[j] = sSim
            insert_index = np.argmax(sim_vec)
            
            #print("Insert index ", insert_index)
            
            g.add_edges([(i, insert_index)])
                
        predicted_labels, gold_labels = [], []
        n_single_clusters = 0.0
        for k_idx, k in enumerate(g.clusters(mode="weak")):
            if len(k) == 1: n_single_clusters += 1.0
            for x in k:
                predicted_labels.append(k_idx)
                gold_labels.append(gold_dict[int(items_list[x].split("::")[1])])

        p, r, f_score = utils.b_cubed(gold_labels,predicted_labels)
        bcubed_prec.append(float(p))
        bcubed_recall.append(float(r))
        bcubed_fscore.append(float(f_score))
        ari = metrics.adjusted_rand_score(gold_labels, predicted_labels)
        ari_vec.append(ari)
        
        n_clusters = len(set(predicted_labels))
        assert(len(set(gold_labels)) == len(set(gold_dict.values())))
        print(gloss, "Socher", str(bcubed_prec[-1]), str(bcubed_recall[-1]), str(bcubed_fscore[-1]), str(ari_vec[-1]), str(len(set(predicted_labels))), str(len(set(gold_labels))), alpha, sep="\t", file=fh)
        
        if args.sample:
            alpha = sample_alpha(n_single_clusters, n_clusters, alpha)
    
    return

def gibbsCRP(pair_dist, gold_dict, gloss):
    """A gibbs sampling based CRP. Does nothing much but moves the items around.
    """
    items_list = list(pair_dist.keys())####Modify this
    assert len(items_list) >= 1

    print("Processing ", gloss)
    alpha = float(args.calpha)
    #alpha = random.expovariate(exp_lambda)
    #Initialize cluster list
    cluster_idx = [[x] for x in items_list]
    n_iter = 1
    clusters_vec, cluster_idxs = [], []
    cluster_idxs.append([x for x in cluster_idx])
    
    ##For the first iteration. Initialize things.
    self_similarity_vec = []
    
    temp_ss_vec = []
    for c in cluster_idx:
        c_sum = 0.0
        for i, j in product(c, c):
            c_sum += pair_dist[i][j]
        temp_ss_vec.append(c_sum)
    
    self_similarity_vec.append(temp_ss_vec[:])

    bcubed_prec, bcubed_recall, bcubed_fscore, ari_vec = [], [], [], []
    previous_cluster_idx = [list(x) for x in cluster_idxs[-1]]
    previous_self_similarity_vec =  self_similarity_vec[-1]
    n_gibbs_step = 0.0
    ari, f_score = 0.0, 0.0

    consensus_clusters = defaultdict(float)

    for n_iter in range(1, max_iter+1):
        #random.shuffle(items_list)
        for n, item in enumerate(items_list):#Find the maximum similar cluster
            n_gibbs_step += 1.0
            cluster_idx = [list(x) for x in previous_cluster_idx]#make a copy of the last cluster idxs. Change it
            for j, j_item in enumerate(cluster_idx):
                if item in j_item:
                    temp_index = j_item.index(item)
                    del cluster_idx[j][temp_index]
                  
            cluster_idx = [x for x in cluster_idx if x != []]##remove empty clusters
            cluster_sum_vec = []#temporary similarity function
            
            if cluster_idx == []:
                cluster_idx.append([item])
                continue
            cluster_sum_vec.append(alpha*pair_dist[item][item])
            #cluster_sum_vec.append(alpha)
            for j_cluster in cluster_idx:
                cluster_sum = 0.0
                for k_j_item in j_cluster:
                    p = pair_dist[item][k_j_item]
                    cluster_sum += p
                cluster_sum_vec.append(cluster_sum)
            #print(item, cluster_sum_vec, sep="\t")
            cluster_sum_vec = np.array(cluster_sum_vec)/np.sum(cluster_sum_vec)

            #insert_index = np.random.choice(range(len(cluster_sum_vec)),p=cluster_sum_vec)
            insert_index = np.argmax(cluster_sum_vec)
            #cluster_idx[insert_index].append(item)
            if insert_index == 0:
                cluster_idx.append([item])
            else:
                cluster_idx[insert_index-1].append(item)
            #print("\t", n_gibbs_step, item, insert_index, cluster_idx[insert_index-1], "\n")
            previous_cluster_idx = [x for x in cluster_idx if x != []]#remove empty clusters
            
            
            
        predicted_labels, gold_labels = [], []
        cluster_idxs.append(previous_cluster_idx)
            
        for k_idx, k in enumerate(previous_cluster_idx):
            for k_item in k:
                predicted_labels.append(int(k_idx))
                gold_labels.append(gold_dict[int(k_item.split("::")[1])])
            for i_item, j_item in combinations(k, r=2):
                consensus_clusters[i_item, j_item] += 1.0
                consensus_clusters[j_item, i_item] += 1.0
                    
        p, r, f_score = utils.b_cubed(gold_labels,predicted_labels)
        
        bcubed_prec.append(float(p))
        bcubed_recall.append(float(r))
        bcubed_fscore.append(float(f_score))
        ari = metrics.adjusted_rand_score(gold_labels, predicted_labels)
        ari_vec.append(ari)
        #print(f_score,ari)
        n_clusters = len(previous_cluster_idx)
        intra_similarity = 0.0
        n_single_clusters = 0.0

        for i, clu in enumerate(previous_cluster_idx):
            if len(clu) == 1:
                n_single_clusters += 1
        
        #print(gloss, str(bcubed_fscore[-1]), str(ari_vec[-1]), str(len(previous_cluster_idx)), str(len(set(gold_labels))), alpha, sep="\t", file=sys.stdout)
        assert(len(set(gold_labels)) == len(set(gold_dict.values())))
        print(gloss, "CRP", str(bcubed_prec[-1]), str(bcubed_recall[-1]), str(bcubed_fscore[-1]), str(ari_vec[-1]), str(len(previous_cluster_idx)), str(len(set(gold_dict.values()))), alpha, sep="\t",file=fh)
        
        if args.sample:
            alpha = sample_alpha(n_single_clusters, n_clusters, alpha)
 
#    print(gloss, *consensus_clustering(consensus_clusters, items_list, gold_dict), sep="\t")
    return

def sample_alpha_log(n_clusts, n_clusters, alpha):
    alpha_new = random.expovariate(exp_lambda)
    mh_ratio = 0.0
    ll_ratio, pr_ratio, hastings_ratio = 0.0, 0.0, 0.0
    ll_ratio = n_clusts*(alpha_new - alpha)
    pr_ratio = -1.0*(alpha_new - alpha)*(exp_lambda+n_clusters)
    hastings_ratio = scaler_alpha*(random.uniform(0,1)-0.5)
    mh_ratio = ll_ratio + pr_ratio + hastings_ratio
    if mh_ratio >= math.log(random.random()):
        return alpha_new
    else:
        return alpha

def sample_alpha_published(n_clusts, n_clusters, alpha):
    """This is the published version of alpha sampling
    """
    alpha_new = random.expovariate(exp_lambda)
    mh_ratio = 0.0
    ll_ratio, pr_ratio, hastings_ratio = 0.0, 0.0, 0.0
    ll_ratio = alpha_new/alpha
    pr_ratio = np.exp(-exp_lambda*(alpha_new-alpha))
    hastings_ratio = np.exp(scaler_alpha*(random.uniform(0,1)-0.5))
    mh_ratio = ll_ratio * pr_ratio * hastings_ratio
    if mh_ratio >= random.random():
        return alpha_new
    else:
        return alpha

def sample_alpha(n_clusts, n_clusters, alpha):
    """This might be the right version of alpha sampling
    """
    if n_clusts == 0:
        return alpha
    alpha_new = random.expovariate(exp_lambda)
    mh_ratio = 0.0
    ll_ratio, pr_ratio, hastings_ratio = 0.0, 0.0, 0.0
    ll_ratio = alpha_new/alpha
    pr_ratio = np.exp(-exp_lambda*(alpha_new-alpha))
    hastings_ratio = np.exp(scaler_alpha*(random.uniform(0,1)-0.5))
    mh_ratio = ll_ratio * pr_ratio * hastings_ratio
    if mh_ratio >= random.random():
        return alpha_new
    else:
        return alpha



def cluster_CRP(d, cogd, fname):
    #f = open('pmi_model/sounds41.txt')
    #sounds = np.array([x.strip() for x in f.readlines()])
    #f.close()

    #f = open('pmi_model/pmi-world.txt','r')
    #l = f.readlines()
    #f.close()
    #logOdds = np.array([x.strip().split() for x in l],np.float)

    #for i in range(len(sounds)):#Initiate sound dictionary
    #    for j in range(len(sounds)):
    #        lodict[sounds[i],sounds[j]] = logOdds[i,j]
    
    fname = fname.split("/")[-1]
    
    print("MEANING","CLUSTER_ALGO","Precision","Recall","F-score","ARI", "# of. Predicted Clusters", "# of. Gold Clusters", "ALPHA", sep="\t", file=fh)
    for gloss in d.keys():
        #if gloss != "I": continue
        #print("Processing ",gloss)
        pair_dist = defaultdict(lambda: defaultdict())

        lex_list = []

        for word_idx in d[gloss].keys():
            word = d[gloss][word_idx]
            lex_list.append(word+"::"+str(word_idx))
        if len(lex_list) < 2: continue
        for sA, sB in product(lex_list, lex_list):
            wordA = sA.split("::")[0]
            wordB = sB.split("::")[0]
            assert len(wordA) > 0
            assert len(wordB) > 0            
            
            score = 0.0
            
            if args.classifier == "nw":
                score = utils.nw(wordA, wordB, scores={}, gop=-2.0, gep=-1.0)[0]
            elif args.classifier == "jaeger":
                score = utils.nw(wordA, wordB, scores=lodict)[0]
            elif args.classifier == "sca":
                score = 1.0-my_SCA(wordA, wordB)
                #if score >=1: print(wordA,wordB, score)
            elif args.classifier == "svm":
               # print(wordA,  wordB, svm_HK.HK(wordA, wordB),sep="\n")
                score = np.dot(HK_test(wordA, wordB, pmi_flag = args.pmi_flag, sca_flag=args.sca_flag),np.array(svm_W))+svm_b
                
            pair_dist[sA][sB] = score
            pair_dist[sB][sA] = score

        simMat = defaultdict(lambda: defaultdict())
        for ka, kb in product(lex_list, lex_list):
#            simMat[ka][kb] = 1.0/(1.0+np.exp(-pair_dist[ka][kb]))
            simMat[ka][kb] = max(0, pair_dist[ka][kb])
        
        gibbsCRP(simMat, cogd[gloss], gloss)
        
        #simMat = defaultdict(lambda: defaultdict())
        #for ka, kb in product(lex_list, lex_list):
        #    simMat[ka][kb] = max(0, pair_dist[ka][kb])
        socherCRP(simMat, cogd[gloss], gloss)
        #continue
        #if args.sample: continue
        
        #simMat = defaultdict(lambda: defaultdict())
        #for ka, kb in product(lex_list, lex_list):
        #    simMat[ka][kb] = max(0, pair_dist[ka][kb])
        eval_scores = utils.upgma(simMat, cogd[gloss], gloss)
        print(gloss,"upgma",*eval_scores,"", sep="\t", file=fh)
        #continue

        distMat = None
        
        if args.classifier == "sca":
            distMat = np.array([[pair_dist[ka][kb] for kb in lex_list] for ka in lex_list])
        else:
            distMat = np.array([[1.0 - (1.0/(1.0+np.exp(-pair_dist[ka][kb]))) for kb in lex_list] for ka in lex_list])
        
        clust = utils.igraph_clustering(distMat, lex_list, threshold=args.thd, method="infomap")
        eval_scores = infomap_eval(clust, cogd[gloss], lex_list)
        print(gloss, "infomap", *eval_scores, "", sep="\t", file=fh)

    return

def infomap_eval(clust, gold_dict, lex_list):
    predicted_labels, gold_labels = [], []
    for k, v in clust.items():
        predicted_labels.append(v)
        gold_labels.append(gold_dict[int(lex_list[k].split("::")[1])])
            
    p, r, f_score = utils.b_cubed(gold_labels,predicted_labels)
    ari = metrics.adjusted_rand_score(gold_labels, predicted_labels)
    eval_scores = [str(p), str(r), str(f_score), str(ari), str(len(set(predicted_labels))), str(len(set(gold_labels)))]
    return eval_scores

fname = args.infile
d, cogd = utils.readCSV(fname)
cluster_CRP(d, cogd, fname)

fh.close()
