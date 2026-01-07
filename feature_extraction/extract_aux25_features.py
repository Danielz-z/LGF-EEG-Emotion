import os
import warnings
import numpy as np
import pywt
from scipy.io import loadmat
from sklearn.linear_model import LinearRegression
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from tqdm import tqdm

electrode_order_names = [
    'FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8',
    'FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8','T7','C5','C3','C1','CZ',
    'C2','C4','C6','T8','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8','P7','P5',
    'P3','P1','PZ','P2','P4','P6','P8','PO7','PO5','PO3','POZ','PO4','PO6','PO8','CB1',
    'O1','OZ','O2','CB2'
]
electrode_names = electrode_order_names
electrode_names_indices = [i for i, x in enumerate(electrode_order_names) if x in electrode_names]

BANDS = [
    (0.5, 3),   # delta
    (3, 7),     # theta
    (7, 13),    # alpha
    (13, 30),   # beta
    (30, 50),   # gamma
]

sf = 200                         # Hz

# windows
WIN_SEC = 4                      # 窗长(s)
STEP_SEC = 4                     # 步长(s)

s1 = ['Happy','Neutral','Disgust','Sad','Anger','Anger','Sad','Disgust','Neutral','Happy']*2
s2 = ['Anger','Sad','Fear','Neutral','Surprise','Surprise','Neutral','Fear','Sad','Anger']*2
s3 = ['Happy','Surprise','Disgust','Fear','Anger','Anger','Fear','Disgust','Surprise','Happy']*2
s4 = ['Disgust','Sad','Fear','Surprise','Happy','Happy','Surprise','Fear','Sad','Disgust']*2
sLabes = [s1, s2, s3, s4]

def labelEmo(strEmo: str) -> int:
    return {'Disgust':0,'Fear':1,'Sad':2,'Neutral':3,'Happy':4,'Anger':5,'Surprise':6}[strEmo]


def compute_wavelet_leaders(signal, wavelet='db4'):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wavelet_obj = pywt.Wavelet(wavelet)
        max_level = min(pywt.dwt_max_level(len(signal), wavelet_obj.dec_len), 6)
        coeffs = pywt.wavedec(signal, wavelet_obj, level=max_level)

    j_exp = np.arange(max_level, 0, -1)
    leaders = []
    for j in range(1, len(coeffs)):
        c_list = []
        for k in range(len(coeffs[j])):
            wind_start = max(0, (k - 1) * (2 ** j_exp[j - 1]))
            wind_end = (k + 2) * (2 ** j_exp[j - 1])
            candidates = []
            for j_inner in range(j, len(coeffs)):
                coefficients = np.abs(coeffs[j_inner])
                time_point = np.arange(len(coefficients)) * (2 ** j_exp[j_inner - 1])
                idx_l = np.where((time_point >= wind_start) & (time_point < wind_end))[0]
                if len(idx_l) > 0:
                    candidates.extend(coefficients[idx_l])
            if candidates:
                c_list.append(np.max(candidates))
        if c_list:
            leaders.append(np.array(c_list))
    return leaders, j_exp

def compute_structure_functions(leaders, j_exp, q_vals=np.linspace(-5, 5, 30)):
    scales = 2 ** j_exp[:len(leaders)]
    S_q = np.zeros((len(q_vals), len(scales)))
    for i, q in enumerate(q_vals):
        for j, l in enumerate(leaders):
            if len(l) == 0:
                S_q[i, j] = np.nan
            elif q == 0:
                S_q[i, j] = np.exp(np.mean(np.log(l[l > 0])))
            else:
                S_q[i, j] = np.nanmean(l ** q)
    valid_cols = ~np.all(np.isnan(S_q), axis=0)
    return S_q[:, valid_cols], scales[valid_cols]

def chhabra_jensen_spectrum(leaders, q_vals, S_q, scales):
    if len(scales) == 0 or S_q.shape[1] == 0:
        n = len(q_vals)
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)

    U = np.zeros((len(leaders), len(q_vals)))
    V = np.zeros((len(leaders), len(q_vals)))
    h_vals, D_vals, z_vals = [], [], []

    for qi, q in enumerate(q_vals):
        j_vals = np.zeros(len(leaders))
        for idx_scale in range(len(leaders)):
            Lx = leaders[idx_scale]
            if len(Lx) == 0:
                continue
            qbic = Lx ** q
            qbic_sum = np.sum(qbic)
            if qbic_sum == 0:
                continue
            qbic = qbic / qbic_sum
            Rx = qbic
            with np.errstate(divide='ignore', invalid='ignore'):
                U[idx_scale, qi] = np.nansum(Rx * np.log2(Rx)) + np.log2(len(Lx))
                V[idx_scale, qi] = np.nansum(Rx * np.log2(Lx))
            j_vals[idx_scale] = len(Lx)

        b = j_vals
        V0, V1, V2 = np.sum(b), np.sum(scales * b), np.sum(scales ** 2 * b)
        denom = V0 * V2 - V1 ** 2
        w = b * (V0 * scales - V1) / denom if denom != 0 else np.zeros_like(b)

        Dq = 1 + np.sum(w * U[:, qi])
        hq = np.sum(w * V[:, qi])
        z_val = np.sum(w * np.log2(S_q[qi, :]))

        D_vals.append(Dq)
        h_vals.append(hq)
        z_vals.append(z_val)

    return np.array(h_vals), np.array(D_vals), np.array(z_vals)

def extract_fractal_features(signal: np.ndarray) -> np.ndarray:
    q_vals = np.linspace(-5, 5, 30)
    feat = np.full(16, np.nan)

    try:
        leaders, j_exp = compute_wavelet_leaders(signal)
        S_q, scales = compute_structure_functions(leaders, j_exp, q_vals)
        h_q, D_q, Z_q = chhabra_jensen_spectrum(leaders, q_vals, S_q, scales)

        valid_z = ~np.isnan(Z_q)
        if np.sum(valid_z) > 1:
            lin = LinearRegression().fit(q_vals[valid_z].reshape(-1, 1), Z_q[valid_z].reshape(-1, 1))
            feat[0] = lin.coef_[0][0]

            X_quad = np.column_stack([q_vals[valid_z] ** 2, q_vals[valid_z], np.ones(np.sum(valid_z))])
            quad = LinearRegression().fit(X_quad, Z_q[valid_z])
            feat[1] = quad.coef_[0]

            feat[2] = np.nanmax(Z_q[valid_z]) - np.nanmin(Z_q[valid_z])
            feat[3] = Z_q[np.nanargmin(np.abs(q_vals - 2))]
            feat[4] = Z_q[np.nanargmin(np.abs(q_vals - 1))]

        if not np.all(np.isnan(h_q)) and not np.all(np.isnan(D_q)):
            idx_sort = np.argsort(h_q)
            D_sorted = D_q[idx_sort]
            if np.any(~np.isnan(D_sorted)):
                idx_max = np.nanargmax(D_sorted)
                left = max(0, idx_max - 5)
                right = min(len(idx_sort), idx_max + 6)
                h_peak = h_q[idx_sort][left:right]
                m = min(len(h_peak), 11)
                feat[5:5 + m] = h_peak[:m]
    except Exception as e:
        print(f"[WARN] fractal feature error: {e}")

    return feat

def extract_time_features(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1 or x.size == 0:
        return np.full(4, np.nan)
    mu = np.nanmean(x)
    sd = np.nanstd(x, ddof=1) if x.size > 1 else np.nan
    sk = skew(x, bias=False, nan_policy='omit') if x.size > 2 else np.nan
    ku = kurtosis(x, fisher=True, bias=False, nan_policy='omit') if x.size > 3 else np.nan
    return np.array([mu, sd, sk, ku], dtype=float)

def band_log_power_welch(signal: np.ndarray, fs: int, bands=BANDS) -> np.ndarray:
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1 or x.size < 8:
        return np.full(len(bands), np.nan)
    nperseg = min(256, x.size)
    noverlap = nperseg // 2
    f, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    feat = np.full(len(bands), np.nan)
    for i, (lo, hi) in enumerate(bands):
        mask = (f >= lo) & (f < hi)
        band_power = np.trapz(Pxx[mask], f[mask]) if np.any(mask) else 0.0
        feat[i] = np.log(max(band_power, 1e-12))  # 避免 log(0)
    return feat

def generateFeatures(testSubject, output_dir):
    dataFile = loadmat("/SEED-VII/EEG_preprocessed/" + str(testSubject) + ".mat")

    features = []
    labels_all = []
    intensities_all = []

    for trialIdx in tqdm(range(1, 81), desc=f"Subject {testSubject}"):
        dataUser = dataFile[str(trialIdx)].T

        sessionId = int(trialIdx / 20)
        videoId = trialIdx % 20
        if videoId == 0:
            videoId = 20
            sessionId -= 1

        emoClass = sLabes[sessionId][videoId - 1]
        labelTrial = labelEmo(emoClass)

        totalSamples = dataUser.shape[0]
        wSizeTime = 4  # seconds
        wSlidingTime = 4  # seconds
        windLength = int(np.ceil(sf * wSizeTime))
        slidingWindow = int(np.ceil(sf * wSlidingTime))
        totalWindows = int((totalSamples - windLength) / slidingWindow) + 1

        for idxWindow in range(totalWindows):
            startW = idxWindow * slidingWindow
            endW = startW + windLength
            segment = dataUser[startW:endW, :]

            featuresWindows = np.empty([62, 25])
            for ch in range(segment.shape[1]):
                f_frac = extract_fractal_features(segment[:, ch])      # 16
                f_time = extract_time_features(segment[:, ch])         # 4
                f_spec = band_log_power_welch(segment[:, ch], fs=sf)   # 5
                featuresWindows[ch, :] = np.concatenate([f_frac, f_time, f_spec])

            features.append(featuresWindows)
            labels_all.append(labelTrial)
            intensities_all.append(0)

        print(f"[Subject {testSubject}] trial {trialIdx}/80 finished, "
              f"accumulative windows: {len(features)}")

    all_features = np.stack(features, axis=0)  # (num_windows, 62, 25)
    labels_all = np.asarray(labels_all)
    intensities_all = np.asarray(intensities_all)

    output_path = os.path.join(output_dir, str(testSubject) + ".npz")
    np.savez_compressed(
        output_path,
        subject_id=testSubject,
        features=all_features,
        labels=labels_all,
        intensities=intensities_all
    )
    print(f"[OK] Subject {testSubject} saved successfully -> {output_path}, "
          f" {all_features.shape[0]} windows in total")

if __name__ == "__main__":
    output_dir = r"/SEED-VII/EEG_preprocessed"
    for subId in range(1, 21):
        generateFeatures(subId, output_dir)
