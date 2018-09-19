#!/bin/bash
python3 pmi_CRP.py -i asjp_data/data-st-64-110.tsv.uniform -o initial_alpha_exps_results/data-st-64-110.HK.pmi.sca.ca.001.results --sample --pmi_flag --sca_flag -ca 0.001&
python3 pmi_CRP.py -i asjp_data/data-st-64-110.tsv.uniform -o initial_alpha_exps_results/data-st-64-110.HK.pmi.sca.ca.01.results --sample --pmi_flag --sca_flag -ca 0.01&
python3 pmi_CRP.py -i asjp_data/data-st-64-110.tsv.uniform -o initial_alpha_exps_results/data-st-64-110.HK.pmi.sca.ca.1.results --sample --pmi_flag --sca_flag -ca 1.0&
wait
python3 pmi_CRP.py -i asjp_data/data-an-45-210.tsv.uniform -o initial_alpha_exps_results/data-an-45-210.HK.pmi.sca.ca.001.results --sample --pmi_flag --sca_flag -ca 0.001&
python3 pmi_CRP.py -i asjp_data/data-an-45-210.tsv.uniform -o initial_alpha_exps_results/data-an-45-210.HK.pmi.sca.ca.01.results --sample --pmi_flag --sca_flag -ca 0.01&
python3 pmi_CRP.py -i asjp_data/data-an-45-210.tsv.uniform -o initial_alpha_exps_results/data-an-45-210.HK.pmi.sca.ca.1.results --sample --pmi_flag --sca_flag -ca 1.0&
wait
python3 pmi_CRP.py -i asjp_data/data-pn-67-183.tsv.uniform -o initial_alpha_exps_results/data-pn-67-183.HK.pmi.sca.ca.001.results --sample --pmi_flag --sca_flag -ca 0.001&
python3 pmi_CRP.py -i asjp_data/data-pn-67-183.tsv.uniform -o initial_alpha_exps_results/data-pn-67-183.HK.pmi.sca.ca.01.results --sample --pmi_flag --sca_flag -ca 0.01&
python3 pmi_CRP.py -i asjp_data/data-pn-67-183.tsv.uniform -o initial_alpha_exps_results/data-pn-67-183.HK.pmi.sca.ca.1.results --sample --pmi_flag --sca_flag -ca 1.0&
wait
python3 pmi_CRP.py -i asjp_data/data-ie-42-208.tsv.uniform -o initial_alpha_exps_results/data-ie-42-208.HK.pmi.sca.ca.001.results --sample --pmi_flag --sca_flag -ca 0.001&
python3 pmi_CRP.py -i asjp_data/data-ie-42-208.tsv.uniform -o initial_alpha_exps_results/data-ie-42-208.HK.pmi.sca.ca.01.results --sample --pmi_flag --sca_flag -ca 0.01&
python3 pmi_CRP.py -i asjp_data/data-ie-42-208.tsv.uniform -o initial_alpha_exps_results/data-ie-42-208.HK.pmi.sca.ca.1.results --sample --pmi_flag --sca_flag -ca 1.0&
wait
python3 pmi_CRP.py -i asjp_data/data-aa-58-200.tsv.uniform -o initial_alpha_exps_results/data-aa-58-200.HK.pmi.sca.ca.001.results --sample --pmi_flag --sca_flag -ca 0.001&
python3 pmi_CRP.py -i asjp_data/data-aa-58-200.tsv.uniform -o initial_alpha_exps_results/data-aa-58-200.HK.pmi.sca.ca.01.results --sample --pmi_flag --sca_flag -ca 0.01&
python3 pmi_CRP.py -i asjp_data/data-aa-58-200.tsv.uniform -o initial_alpha_exps_results/data-aa-58-200.HK.pmi.sca.ca.1.results --sample --pmi_flag --sca_flag -ca 1.0&
wait
