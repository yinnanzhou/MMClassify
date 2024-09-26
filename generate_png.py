# spec_mmwave：毫米波的stft结果，采样率为16000hz；
# spec_audio：语音的stft结果，采样率为16000hz；
# columns：未说话片段时间索引；
# spec_noise : 噪音的stft结果，采样率为16000hz。

import h5py
import torch

file_path = r'/home/mambauser/MMClassify/data/data_processed_noise_mma.hdf5'

preloaded_data = {}
preloaded_noise = {}
with h5py.File(file_path, 'r') as hdf5_file:
    group_names = [i for i in hdf5_file.keys() if i != 'various_noise']
    noise_list = [
        i[11:] for i in hdf5_file['various_noise'].keys()
        if 'spec_NOISE_' in i
    ]
    for group_name in group_names:
        try:
            preloaded_data[group_name] = {
                'spec_mmwave':
                torch.from_numpy(
                    hdf5_file[group_name]['spec_mmwave'][:]).to(
                        torch.complex64),
                'spec_audio':
                torch.from_numpy(
                    hdf5_file[group_name]['spec_audio'][:]).to(
                        torch.complex64),
                'columns':
                hdf5_file[group_name]['columns'][:]
            }
        except KeyError as e:
            print(f"KeyError for group {group_name}: {e}")
    preloaded_noise = {
        noise: {
            'spec_noise':
            torch.from_numpy(
                hdf5_file['various_noise'][f'spec_NOISE_{noise}'][:])
        }
        for noise in noise_list
    }
    
    
from MMGenerateFunc import MMGenerateFunctions, MMPlot
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

save_path = r"/home/mambauser/MMClassify/data/dataPng/continuous_5_300"


# 找到每段连续说话的第一个索引（针对新Zxx_speaking）
def find_continuous_segments_new(Zxx_speaking, indices):
    if not indices:
        return []

    first_indices = [0]  # 新的索引中的第一个元素

    for i in range(1, len(indices)):
        if indices[i] != indices[i - 1] + 1:
            first_indices.append(i)

    first_indices.append(Zxx_speaking.shape[1])

    return first_indices

# # 将每个group的处理逻辑封装成函数
# def process_group(group_name):
#     Zxx = preloaded_data[group_name]["spec_mmwave"][5:300, :]
#     parts = group_name.split("_")
#     wordIndex = 0 if parts[0] == "google" else 1 if parts[0] == "timit" else None
#     personIndex = int(parts[-2])
#     repeatIndex = int(parts[-1])

#     silent_indices = preloaded_data[group_name]["columns"]

#     # 说话的时间索引
#     speaking_indices = sorted(set(range(Zxx.shape[1])) - set(silent_indices))

#     # 筛选说话的片段
#     Zxx_speaking = Zxx[:, speaking_indices]

#     # 使用新的索引列表来调用函数
#     first_indices = find_continuous_segments_new(Zxx_speaking, speaking_indices)

#     MMPlot.saveSTFT_hdf5_mmwave(
#         Zxx=Zxx,
#         first_indices=first_indices,
#         save_path=save_path,
#         wordIndex=wordIndex,
#         personIndex=personIndex,
#         repeatIndex=repeatIndex,
#     )

# # 使用ThreadPoolExecutor并行处理
# with ThreadPoolExecutor() as executor:
#     list(tqdm(executor.map(process_group, group_names), total=len(group_names), desc="Processing Groups"))


for group_name in tqdm(group_names, desc="Processing Groups"):
    Zxx = preloaded_data[group_name]["spec_mmwave"][5:300, :]
    # t = np.arange(0, (Zxx.shape[1]) * 0.018, 0.018)
    # f = np.linspace(0, 16000 * Zxx.shape[0] / 481, num=Zxx.shape[0])
    # MMPlot.plotSTFT(t, f, Zxx)
    # MMPlot.plotSTFT_new(t, f, Zxx, 10, 60)

    parts = group_name.split("_")
    wordIndex = 0 if parts[0] == "google" else 1 if parts[0] == "timit" else none
    personIndex = int(parts[-2])
    repeatIndex = int(parts[-1])

    silent_indices = preloaded_data[group_name]["columns"]

    # 说话的时间索引
    speaking_indices = sorted(set(range(Zxx.shape[1])) - set(silent_indices))

    # 筛选说话的片段
    Zxx_speaking = Zxx[:, speaking_indices]

    # 使用新的索引列表来调用函数
    first_indices = find_continuous_segments_new(Zxx_speaking, speaking_indices)

    MMPlot.saveSTFT_hdf5_mmwave(
        Zxx=Zxx,
        first_indices=first_indices,
        save_path=save_path,
        wordIndex=wordIndex,
        personIndex=personIndex,
        repeatIndex=repeatIndex,
    )