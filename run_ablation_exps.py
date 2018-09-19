import glob, subprocess
import itertools as it

alpha = str(0.1)
datasets = ["asjp_data/central_asian.tsv.uniform",  "asjp_data/abvd2.tsv.asjp","asjp_data/IELex-2016.tsv.asjp", "asjp_data/Austro-Asiatic-122.tsv.uniform"]

new_data = glob.glob("asjp_data/data-*")

#datasets = ["asjp_data/central_asian.tsv.uniform", "asjp_data/abvd2.tsv.asjp", "asjp_data/IELex-2016.tsv.asjp", "asjp_data/Austro-Asiatic-122.tsv.uniform", "data-aa-58-200.tsv.uniform", "data-an-45-210.tsv.uniform", ]

print("#!/bin/bash")
#for dataset in new_data:
for dataset in datasets:
#    if "-ie-" in dataset: continue
#    print(dataset, sep="\t")
    outfile = "results_240518/"+dataset.split("/")[-1].split(".")[0]
#    subprocess.run(["python3", "pmi_CRP.py", "-i", dataset, "-o", outfile+".HK.results", "--sample"])
#    subprocess.run(["python3", "pmi_CRP.py", "-i", dataset, "-o", outfile+".HK.pmi.results", "--sample", "--pmi_flag"])
#    subprocess.run(["python3", "pmi_CRP.py", "-i", dataset, "-o", outfile+".HK.sca.results", "--sample", "--sca_flag"])    
#    subprocess.run(["python3", "pmi_CRP.py", "-i", dataset, "-o", outfile+".HK.pmi.sca.results", "--sample", "--pmi_flag", "--sca_flag"])

    print("python3", "pmi_CRP.py", "-i", dataset, "-o", outfile+".HK.results", "--sample", "&", sep=" ")
    print("python3", "pmi_CRP.py", "-i", dataset, "-o", outfile+".HK.pmi.results", "--sample", "--pmi_flag", "&", sep=" ")
    print("wait")
    print("python3", "pmi_CRP.py", "-i", dataset, "-o", outfile+".HK.sca.results", "--sample", "--sca_flag", "&", sep=" ")    
    print("python3", "pmi_CRP.py", "-i", dataset, "-o", outfile+".HK.pmi.sca.results", "--sample", "--pmi_flag", "--sca_flag", "&", sep=" ")
    print("wait")
