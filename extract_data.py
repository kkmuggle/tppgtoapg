import os
import pickle
import h5py
from tqdm import tqdm
from scipy.signal import detrend
import numpy as np
import matplotlib as plt

def sliding_window(data, window_size, overlap):
    step = int(window_size * (1 - overlap))
    segments = []
    for i in range(0, len(data) - window_size + 1, step):
        segments.append(data[i:i + window_size])
    return np.array(segments)

window_size = 1024
overlap = 0.75


try:
    os.makedirs('train')
    os.makedirs('predict')
except Exception as e:
    print(e)

length = 125 * 60 * 8   # 提取8min以上信号

for k in range(1, 3):


    print("Extract Part_{}".format(k))

    f = h5py.File(os.path.join('raw_data', 'Part_{}.mat'.format(k)))
    ky = "Part_" + str(k)

    for i in tqdm(range(len(f[ky])), desc='Extract Cells'):

        ppg = []
        abp = []
        ecg = []

        if len(f[f[ky][i][0]]) >= length:

            for ii in tqdm(range(len(f[f[ky][i][0]])), desc='Extract Cell_{}'.format(i+1)):

                ppg.append(f[f[ky][i][0]][ii][0])
                abp.append(f[f[ky][i][0]][ii][1])
                ecg.append(f[f[ky][i][0]][ii][2])

            if any(x > 200 for x in abp):   # 筛选血压在200以下数据
                continue
            else:
                ppg = np.array(ppg)
                abp = np.array(abp)
                ecg = np.array(ecg)

                ppg_detrend = detrend(ppg)  # PPG信号进行detrend预处理
                ppg_segments = sliding_window(ppg_detrend, window_size, overlap)
                abp_segments = sliding_window(abp, window_size, overlap)
                ecg_segments = sliding_window(ecg, window_size, overlap)

                dataset = np.stack((ppg_segments, abp_segments, ecg_segments), axis=-1)

                train_idx = int(len(dataset) * 0.8)

                pickle.dump(dataset[:train_idx], open(os.path.join('train', '{}_{}.p'.format(k, i+1)), 'wb'))
                pickle.dump(dataset[train_idx:], open(os.path.join('predict', '{}_{}.p'.format(k, i+1)), 'wb'))

        else:
            continue