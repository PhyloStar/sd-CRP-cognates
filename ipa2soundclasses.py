from collections import defaultdict
import sys
from lingpy import *

ipa2asjp_dict = defaultdict(str)
lines = open("ipa2asjp.txt", "r").readlines()
lines = [line.replace("\n", "") for line in lines]
lines = [line.replace("\r", "") for line in lines]
lines = [line.split(" ") for line in lines]
for ipa, asjp in lines:
    ipa2asjp_dict[ipa] = asjp

def convert(w):
    x = ""
    for ch in w:
        x += ipa2asjp_dict[ch]
    return x

print("Language","Concept","IPA","ASJP","SCA",sep="\t")
for line in open(sys.argv[1]):
    line = line.replace("\n", "")
    line = line.replace(" ", "").split("\t")
    lang, concept, word, cc = line[0], line[2], line[5], line[6]
    asjp_words, sca_words = [], []
    for w in word.split(" "):
        w = w.replace(" ", "")
        w = w.replace("(", "")
        w = w.replace(")", "")
        for ch in w:
            if ch not in ipa2asjp_dict: 
                print(ch, " in ", line)
                sys.exit(1)
             
    for w in word.split(","):
        if w.startswith("-"):
            asjp_words += [w]
            sca_words += [w]
        else:
            asjp_words += [convert(w)]
            sca_words += ["".join(tokens2class(ipa2tokens(w), 'sca')).replace("0","")]
        
    print(lang+"\t"+concept+"\t"+word+"\t"+", ".join(asjp_words)+"\t"+", ".join(sca_words))
