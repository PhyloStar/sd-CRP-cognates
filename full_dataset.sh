#!/bin/bash
python3 pmi_CRP.py -i asjp_data/central_asian.tsv.uniform -o results_240518/central_asian.HK.results --sample &
python3 pmi_CRP.py -i asjp_data/central_asian.tsv.uniform -o results_240518/central_asian.HK.pmi.results --sample --pmi_flag &
wait
python3 pmi_CRP.py -i asjp_data/central_asian.tsv.uniform -o results_240518/central_asian.HK.sca.results --sample --sca_flag &
python3 pmi_CRP.py -i asjp_data/central_asian.tsv.uniform -o results_240518/central_asian.HK.pmi.sca.results --sample --pmi_flag --sca_flag &
wait
python3 pmi_CRP.py -i asjp_data/abvd2.tsv.asjp -o results_240518/abvd2.HK.results --sample &
python3 pmi_CRP.py -i asjp_data/abvd2.tsv.asjp -o results_240518/abvd2.HK.pmi.results --sample --pmi_flag &
wait
python3 pmi_CRP.py -i asjp_data/abvd2.tsv.asjp -o results_240518/abvd2.HK.sca.results --sample --sca_flag &
python3 pmi_CRP.py -i asjp_data/abvd2.tsv.asjp -o results_240518/abvd2.HK.pmi.sca.results --sample --pmi_flag --sca_flag &
wait
python3 pmi_CRP.py -i asjp_data/IELex-2016.tsv.asjp -o results_240518/IELex-2016.HK.results --sample &
python3 pmi_CRP.py -i asjp_data/IELex-2016.tsv.asjp -o results_240518/IELex-2016.HK.pmi.results --sample --pmi_flag &
wait
python3 pmi_CRP.py -i asjp_data/IELex-2016.tsv.asjp -o results_240518/IELex-2016.HK.sca.results --sample --sca_flag &
python3 pmi_CRP.py -i asjp_data/IELex-2016.tsv.asjp -o results_240518/IELex-2016.HK.pmi.sca.results --sample --pmi_flag --sca_flag &
wait
python3 pmi_CRP.py -i asjp_data/Austro-Asiatic-122.tsv.uniform -o results_240518/Austro-Asiatic-122.HK.results --sample &
python3 pmi_CRP.py -i asjp_data/Austro-Asiatic-122.tsv.uniform -o results_240518/Austro-Asiatic-122.HK.pmi.results --sample --pmi_flag &
wait
python3 pmi_CRP.py -i asjp_data/Austro-Asiatic-122.tsv.uniform -o results_240518/Austro-Asiatic-122.HK.sca.results --sample --sca_flag &
python3 pmi_CRP.py -i asjp_data/Austro-Asiatic-122.tsv.uniform -o results_240518/Austro-Asiatic-122.HK.pmi.sca.results --sample --pmi_flag --sca_flag &
