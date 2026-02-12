# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:50:00 2023

@author: psp
"""
import os
import openpyxl
import numpy as np
import pandas as pd
import re
import pitch_hist_eval
import pitch_hist_band_eval
import pitch_entropy
from scipy.special import expit

class listening_result():
    def __init__(self, name, age):
        self.name = name
        self.age = age
        self.result = []

def parse_sheet(target_folder):

    rank_list = []
    rank_expert_list = []
    folder_path = target_folder + '/sub_score_list'

    # Loop through each file in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is an .xlsx file
        if filename.endswith(".xlsx"):
            # Full path to the file
            file_path = os.path.join(folder_path, filename)
            print(f'parsing {file_path}...')
            
            # Open the workbook
            workbook = openpyxl.load_workbook(file_path)
            
            # Assuming the data is in the first worksheet
            sheet = workbook.active
            single_result = listening_result(name=sheet[f"B{1}"].value, age=sheet[f"B{2}"].value)

            # Loop through rows 4 to 65 (indices start from 1 in openpyxl)
            # Column to check (A = 1, B = 2, etc.)
            for row in sheet.iter_rows(min_col=2, max_col=2, min_row=4, max_row=65):
                # cell_value = sheet[f"B{row}"].value
                cell_value = row[0].value
                # single_result.result[row - 4] = cell_value
                if cell_value is not None:
                    single_result.result.append(cell_value)
                    print(f'{row[0]}: {cell_value}')

            rank_list.append(single_result.result)

            if single_result.age is not None:
                # print(f'single_result.age = {type(single_result.age)}')
                if not isinstance(single_result.age, int):
                    if isinstance(single_result.age, float):
                        single_result.age = str(int(single_result.age))
                    age = int(re.findall(r'\d+', single_result.age)[0])
                if age >= 1:
                    rank_expert_list.append(single_result.result)

    print(f'rank_list shape: {np.shape(rank_list)}')
    print(f'median rank_list shape: {np.shape(np.median(rank_list, axis=0))}')
    print(f'rank_expert_list shape: {np.shape(rank_expert_list)}')
    print(f'median rank_expert_list shape: {np.shape(np.median(rank_expert_list, axis=0))}')

    return np.median(rank_list, axis=0), np.median(rank_expert_list, axis=0)
            

def compute_obj_score(target_folder):
    # Analyze pitch output from model_v8
    gmm_peakBW_list = []
    entropy_list = []
    predict_folder = target_folder + '/predict_result'
    files = os.listdir(predict_folder)
    # sorted_files = sorted(files, key=lambda x: int(x.split('_')[0]))
    sorted_files = sorted(files, key=lambda x: int(re.findall(r'\d+', x)[0]))

    for filename in sorted_files:
        filepath = os.path.join(predict_folder, filename)
        print(f'analyzing {filepath}...')

        # peakBW score
        single_gmm_peakBW_result = pitch_hist_eval.gmm_peakBW_eval(filepath, is_segment=True)
        single_peakBW = single_gmm_peakBW_result.avg_peakBW_measure
        print(f'peakBW measure: {single_peakBW}')
        gmm_peakBW_list.append(single_peakBW)

        # Entropy
        single_entropy_result = pitch_entropy.get_entropy(filepath, is_segment=True)
        entropy_list.append(single_entropy_result)
        print(f'entropy: {single_entropy_result}\n')

        # # band-wise peakBW score
        # single_gmm_peakBW_result_band = pitch_hist_band_eval.gmm_peakBW_band_eval(filepath)
        # single_peakBW_band = single_gmm_peakBW_result_band.avg_peakBW_measure
        # print(f'band-wise peakBW measure: {single_peakBW_band}')
        # gmm_peakBW_list_band.append(single_peakBW_band)

    # normalized peakBW score
    gmm_peakBW_list = np.asarray(gmm_peakBW_list)
    # gmm_peakBW_list = (gmm_peakBW_list - np.mean(gmm_peakBW_list)) / np.std(gmm_peakBW_list)
    # gmm_peakBW_list = 3*expit(gmm_peakBW_list) +1 # Apply sigmoid and rescale: 1~4 levels

    return gmm_peakBW_list, entropy_list


if __name__ == '__main__':

    save_folder = 'listening_test/'
    # Get sub score from xlsx sheet
    rank_arr_1, expert_rank_arr_1 = parse_sheet('listening_test/first')
    rank_arr_2, expert_rank_arr_2 = parse_sheet('listening_test/second')
    rank_arr = np.concatenate((rank_arr_1, rank_arr_2), axis=0)
    expert_rank_arr = np.concatenate((expert_rank_arr_1, expert_rank_arr_2), axis=0)

    np.savetxt(save_folder+'rank_medians.txt', rank_arr)
    # np.savetxt(save_folder+'expert_rank_medians.txt', expert_rank_arr)
    print(f'sub. score saved in {save_folder}.')

    # # Get obj score
    # gmm_arr_1, entropy_arr_1 = compute_obj_score('listening_test/first')
    # gmm_arr_2, entropy_arr_2 = compute_obj_score('listening_test/second')
    # gmm_arr = np.concatenate((gmm_arr_1, gmm_arr_2), axis=0)
    # entropy_arr = np.concatenate((entropy_arr_1, entropy_arr_2), axis=0)

    # np.savetxt(save_folder+'gmm_scores_seg.txt', gmm_arr)
    # np.savetxt(save_folder+'entropy_scores_seg_unroll.txt', entropy_arr)
    # print(f'obj. score saved in {save_folder}.')