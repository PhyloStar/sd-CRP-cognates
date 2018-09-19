#!/bin/bash
python3 pmi_CRP.py -i asjp_data/data-st-64-110.tsv.uniform -o results_240518/data-st-64-110.HK.results --sample &
python3 pmi_CRP.py -i asjp_data/data-st-64-110.tsv.uniform -o results_240518/data-st-64-110.HK.pmi.results --sample --pmi_flag &
wait
python3 pmi_CRP.py -i asjp_data/data-st-64-110.tsv.uniform -o results_240518/data-st-64-110.HK.sca.results --sample --sca_flag &
python3 pmi_CRP.py -i asjp_data/data-st-64-110.tsv.uniform -o results_240518/data-st-64-110.HK.pmi.sca.results --sample --pmi_flag --sca_flag &
wait
python3 pmi_CRP.py -i asjp_data/data-an-45-210.tsv.uniform -o results_240518/data-an-45-210.HK.results --sample &
python3 pmi_CRP.py -i asjp_data/data-an-45-210.tsv.uniform -o results_240518/data-an-45-210.HK.pmi.results --sample --pmi_flag &
wait
python3 pmi_CRP.py -i asjp_data/data-an-45-210.tsv.uniform -o results_240518/data-an-45-210.HK.sca.results --sample --sca_flag &
python3 pmi_CRP.py -i asjp_data/data-an-45-210.tsv.uniform -o results_240518/data-an-45-210.HK.pmi.sca.results --sample --pmi_flag --sca_flag &
wait
python3 pmi_CRP.py -i asjp_data/data-pn-67-183.tsv.uniform -o results_240518/data-pn-67-183.HK.results --sample &
python3 pmi_CRP.py -i asjp_data/data-pn-67-183.tsv.uniform -o results_240518/data-pn-67-183.HK.pmi.results --sample --pmi_flag &
wait
python3 pmi_CRP.py -i asjp_data/data-pn-67-183.tsv.uniform -o results_240518/data-pn-67-183.HK.sca.results --sample --sca_flag &
python3 pmi_CRP.py -i asjp_data/data-pn-67-183.tsv.uniform -o results_240518/data-pn-67-183.HK.pmi.sca.results --sample --pmi_flag --sca_flag &
wait
python3 pmi_CRP.py -i asjp_data/data-aa-58-200.tsv.uniform -o results_240518/data-aa-58-200.HK.results --sample &
python3 pmi_CRP.py -i asjp_data/data-aa-58-200.tsv.uniform -o results_240518/data-aa-58-200.HK.pmi.results --sample --pmi_flag &
wait
python3 pmi_CRP.py -i asjp_data/data-aa-58-200.tsv.uniform -o results_240518/data-aa-58-200.HK.sca.results --sample --sca_flag &
python3 pmi_CRP.py -i asjp_data/data-aa-58-200.tsv.uniform -o results_240518/data-aa-58-200.HK.pmi.sca.results --sample --pmi_flag --sca_flag &
wait
python3 pmi_CRP.py -i asjp_data/data-ie-42-208.tsv.uniform -o results_240518/data-ie-42-208.HK.results --sample &
python3 pmi_CRP.py -i asjp_data/data-ie-42-208.tsv.uniform -o results_240518/data-ie-42-208.HK.pmi.results --sample --pmi_flag &
wait
python3 pmi_CRP.py -i asjp_data/data-ie-42-208.tsv.uniform -o results_240518/data-ie-42-208.HK.sca.results --sample --sca_flag &
python3 pmi_CRP.py -i asjp_data/data-ie-42-208.tsv.uniform -o results_240518/data-ie-42-208.HK.pmi.sca.results --sample --pmi_flag --sca_flag &
