# sd-CRP-cognates
Similarity dependent CRP

The folder asjp_data contains the data required to train the SVM model. Training is conducted using the following files: "afrasian.tsv", "Bai-110-09.tsv.asjp", "kamasau.tsv", "miao_yao.tsv","mayan.tsv","kadai.tsv","mixe_zoque.tsv","ObUgrian-110-21.tsv.asjp",  "Japanese-200-10.tsv.asjp", "mon_khmer.tsv", "lolo_burmese.tsv"

The full datasets are:
"asjp_data/central_asian.tsv.uniform",  "asjp_data/abvd2.tsv.asjp","asjp_data/IELex-2016.tsv.asjp", "asjp_data/Austro-Asiatic-122.tsv.uniform"

The pruned datasets are:
"asjp_data/data-aa-58-200.tsv.uniform", "asjp_data/data-ie-42-208.tsv.uniform", "asjp_data/data-st-64-110.tsv.uniform", "asjp_data/data-an-45-210.tsv.uniform", "asjp_data/data-pn-67-183.tsv.uniform"

The SVM model can be trained by running the svm_HK.py program.

The clustering results are generated by running the following command:

`python3 pmi_CRP.py -i dataset -o outfile --sample`

By default, the program runs all the four algorithms ns-CRP, sb-CRP, InfoMap and UPGMA.

`python3 pmi_CRP.py -h` would output the following:

```2018-05-30 17:48:07,997 [INFO] Successfully changed parameters.
usage: pmi_CRP.py [-h] [-ca CALPHA] [-t THD] [-i INFILE] [-o OUTFILE]
                  [-A IN_ALPHABET] [-clf CLASSIFIER] [-w WEIGHTS] [--pmi_flag]
                  [--sca_flag] [--sample]

CRP for word lists

optional arguments:
  -h, --help            show this help message and exit
  -ca CALPHA, --calpha CALPHA
                        CRP alpha
  -t THD, --thd THD     A number between 0 and 1 for clustering
  -i INFILE, --infile INFILE
                        input file name
  -o OUTFILE, --outfile OUTFILE
                        output file name
  -A IN_ALPHABET, --in_alphabet IN_ALPHABET
                        input alphabet asjp, sca, dolgo
  -clf CLASSIFIER, --classifier CLASSIFIER
                        jaeger PMI, svm, nw, sca
  -w WEIGHTS, --weights WEIGHTS
                        SVM weights file name
  --pmi_flag            add PMI
  --sca_flag            add SCA
  --sample              sample crp alpha

```

Requirements: LingPy which can be installed as

`pip install lingpy`


