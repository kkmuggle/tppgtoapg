import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


def reconstruct_signal(segments, window_size, overlap):
    step = int(window_size * (1 - overlap))
    signal_length = len(segments) * step + window_size - step
    reconstructed_signal = np.zeros(signal_length)

    for i, segment in enumerate(segments):
        start = i * step
        end = start + window_size

        segment_length = int(window_size * 0.25)

        if i == len(segments) - 1:
            reconstructed_signal[start:end] = segment
        else:
            reconstructed_signal[start:start + segment_length] = segment[:segment_length]

    return reconstructed_signal


def extract_sbp_dbp_map(segment):
    sbp = np.max(segment)
    dbp = np.min(segment)
    map = np.mean(segment)
    return sbp, dbp, map


sbp_dbp_map_list = []
tei_list = []

predict_data_folder = r"G:\MYPROJECT\1\experiment\subject_dependent\subject_depend_predict_data_u2net"

predict_files = [file for file in os.listdir(predict_data_folder) if
                 file.startswith('predictions_and_ground_truth_') and file.endswith('.p')]
file_index = 1

for predict_file in tqdm(sorted(predict_files)):
    with open(os.path.join(predict_data_folder, predict_file), "rb") as f:
        predictions, ground_truths = pickle.load(f)

    predictions = [pred.cpu().numpy().reshape(-1) for pred in predictions]
    ground_truths = [gt.cpu().numpy().reshape(-1) for gt in ground_truths]

    window_size = 1024
    overlap = 0.75

    reconstructed_predictions = reconstruct_signal(np.vstack(predictions), window_size, overlap) * 200
    reconstructed_ground_truths = reconstruct_signal(np.vstack(ground_truths), window_size, overlap) * 200

    sbp_dbp_map_values = []
    sbp_preds, dbp_preds, sbp_gts, dbp_gts = [], [], [], []

    dtw_values = []  # Store all DTW values

    for preds, gts in zip(predictions, ground_truths):
        sbp_pred, dbp_pred, map_pred = extract_sbp_dbp_map(preds)
        sbp_gt, dbp_gt, map_gt = extract_sbp_dbp_map(gts)

        sbp_dbp_map_values.append(
            (sbp_pred * 200, dbp_pred * 200, map_pred * 200, sbp_gt * 200, dbp_gt * 200, map_gt * 200))

        sbp_preds.append(sbp_pred)
        dbp_preds.append(dbp_pred)
        sbp_gts.append(sbp_gt)
        dbp_gts.append(dbp_gt)

        dtw, _ = fastdtw(gts * 200, preds * 200, dist=euclidean)
        max_possible_distance = len(gts) * np.abs(np.max(gts * 200) - np.min(gts * 200))
        normalized_dtw = dtw / max_possible_distance  # Normalize DTW
        dtw_values.append(normalized_dtw)  # Store normalized DTW

    sbp_preds = np.array(sbp_preds)
    dbp_preds = np.array(dbp_preds)
    sbp_gts = np.array(sbp_gts)
    dbp_gts = np.array(dbp_gts)

    sbp_nrmse = np.sqrt(mean_squared_error(sbp_gts, sbp_preds)) / (sbp_gts.max()-sbp_gts.min())
    dbp_nrmse = np.sqrt(mean_squared_error(dbp_gts, dbp_preds)) / (dbp_gts.max()-dbp_gts.min())

    mean_dtw = np.mean(dtw_values)  # Compute the mean of all DTW values

    tei = np.sqrt((sbp_nrmse ** 2 + dbp_nrmse ** 2 + mean_dtw ** 2) / 3)
    tei_list.append(tei)

    sbp_dbp_map_df = pd.DataFrame(
        sbp_dbp_map_values,
        columns=['SBP_Pred', 'DBP_Pred', 'MAP_Pred', 'SBP_GT', 'DBP_GT', 'MAP_GT'])

    file_index += 1

tei_array = np.array(tei_list)
tei_mae = np.mean(np.abs(tei_array - np.mean(tei_array)))
tei_std = np.std(tei_array)

print('TEI MAE: ', tei_mae)
print('TEI STD: ', tei_std)