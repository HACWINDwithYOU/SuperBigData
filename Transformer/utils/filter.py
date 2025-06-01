import numpy as np
from scipy.signal import savgol_filter, butter, filtfilt

def moving_average(data, window_length=5):
    """
    对数据进行移动平均滤波
    参数:
        data: 输入数据 (1D 数组)
        window_size: 滑动窗口大小
    返回:
        滤波后的数据
    """
    kernel = np.ones(window_length)/window_length
    convolved_matrix = np.array([
        np.convolve(data[:, i], kernel, mode='same') for i in range(data.shape[1])
    ]).T
    return convolved_matrix

def apply_savgol_filter(data, window_length=11, polyorder=2):
    """
    对数据应用 Savitzky-Golay 滤波
    参数:
        data: 输入数据 (1D 数组)
        window_length: 滑动窗口长度，必须为奇数
        polyorder: 多项式阶数
    返回:
        滤波后的数据
    """
    return savgol_filter(data, window_length=window_length, polyorder=polyorder, axis=0)

def butter_lowpass_filter(data, cutoff, fs, order=4):
    """
    对数据应用 Butterworth 低通滤波
    参数:
        data: 输入数据 (1D 数组)
        cutoff: 截止频率
        fs: 采样频率
        order: 滤波器阶数
    返回:
        滤波后的数据
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)