import numpy as np
import pywt
import os
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import warnings


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
        return np.full(len(q_vals), np.nan), np.full(len(q_vals), np.nan), np.full(len(q_vals), np.nan)

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
        if denom != 0:
            w = b * (V0 * scales - V1) / denom
        else:
            w = np.zeros_like(b)

        Dq = 1 + np.sum(w * U[:, qi])
        hq = np.sum(w * V[:, qi])
        z_val = np.sum(w * np.log2(S_q[qi, :]))

        D_vals.append(Dq)
        h_vals.append(hq)
        z_vals.append(z_val)

    return np.array(h_vals), np.array(D_vals), np.array(z_vals)


def extract_fractal_features(signal):
    q_vals = np.linspace(-5, 5, 30)

    try:
        leaders, j_exp = compute_wavelet_leaders(signal)
        S_q, scales = compute_structure_functions(leaders, j_exp, q_vals)
        h_q, D_q, Z_q = chhabra_jensen_spectrum(leaders, q_vals, S_q, scales)

        features = np.full(16, np.nan)

        if not np.all(np.isnan(Z_q)):
            valid_z = ~np.isnan(Z_q)
            if np.sum(valid_z) > 1:
                linear_model = LinearRegression().fit(
                    q_vals[valid_z].reshape(-1, 1),
                    Z_q[valid_z].reshape(-1, 1))
                features[0] = linear_model.coef_[0][0]

            X_quad = np.column_stack([q_vals[valid_z] ** 2, q_vals[valid_z], np.ones(np.sum(valid_z))])
            quad_model = LinearRegression().fit(X_quad, Z_q[valid_z])
            features[1] = quad_model.coef_[0]

            features[2] = np.nanmax(Z_q) - np.nanmin(Z_q)
            features[3] = Z_q[np.argmin(np.abs(q_vals - 2))]
            features[4] = Z_q[np.argmin(np.abs(q_vals - 1))]

            if not np.all(np.isnan(h_q)):
                idx_sort = np.argsort(h_q)
                valid_d = ~np.isnan(D_q[idx_sort])
                if np.any(valid_d):
                    idx_max = np.nanargmax(D_q[idx_sort])
                    h_peak = h_q[idx_sort][max(0, idx_max - 5):idx_max + 6]
                    features[5:5 + len(h_peak)] = h_peak

    except Exception as e:
        print(f"Error in feature extraction: {str(e)}")
        features = np.full(16, np.nan)

    return features


def parse_filename(filename):
    base = os.path.basename(filename)
    try:
        subject_id = int(base.split('_')[0].split('.')[0])
        return subject_id
    except:
        return -1


def process_npz_file(npz_path, output_dir):
    try:
        data = np.load(npz_path, allow_pickle=True)
        filtered_data = data['filteredData']

        label = data['label'].item() if 'label' in data else -1
        intensity = data['intensity'] if 'intensity' in data else np.full(filtered_data.shape[0], -1)

        subject_id = parse_filename(npz_path)
        n_segments = filtered_data.shape[0]

        all_features = []
        valid_segments = 0

        for i in range(n_segments):
            segment = filtered_data[i].T
            if segment.shape[0] < 100:
                continue

            features = []
            for ch in range(segment.shape[1]):
                ch_features = extract_fractal_features(segment[:, ch])
                features.append(ch_features)

            if len(features) == 62:
                all_features.append(np.array(features))
                valid_segments += 1

        if valid_segments > 0:
            output_path = os.path.join(output_dir, os.path.basename(npz_path))
            np.savez_compressed(
                output_path,
                subject_id=np.full(valid_segments, subject_id, dtype=int),
                features=np.array(all_features),  # (n_valid_segments, 62, 16)
                labels=np.full(valid_segments, label, dtype=int),
                intensities=intensity[:valid_segments] if len(intensity) >= valid_segments else np.full(valid_segments,
                                                                                                        -1)
            )
            return True
        return False

    except Exception as e:
        print(f"Error processing {os.path.basename(npz_path)}: {str(e)}")
        return False


def batch_process(input_dir, output_dir):
    input_dir = os.path.normpath(input_dir)
    output_dir = os.path.normpath(output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = [f for f in os.listdir(input_dir) if f.endswith('.npz')]
    success_count = 0

    for file in tqdm(files, desc='Processing files'):
        npz_path = os.path.join(input_dir, file)
        if process_npz_file(npz_path, output_dir):
            success_count += 1

    print(f"\nProcessing complete. Successfully processed {success_count}/{len(files)} files.")


def load_all_features(output_dir):
    output_dir = os.path.normpath(output_dir)
    files = [f for f in os.listdir(output_dir) if f.endswith('.npz')]

    features, labels, intensities, subject_ids = [], [], [], []

    for file in files:
        try:
            data = np.load(os.path.join(output_dir, file), allow_pickle=True)
            if 'features' in data:
                feats = data['features']                # (n_segments, 62, 16)
                labs  = data['labels']                  # (n_segments,)
                ints  = data['intensities']             # (n_segments,)
                n_segments = feats.shape[0]

                sid = data['subject_id']
                if np.ndim(sid) == 0:
                    sid = np.full((n_segments,), int(sid))
                else:
                    if len(sid) != n_segments:
                        sid = np.full((n_segments,), int(sid[0]))

                features.append(feats)
                labels.append(labs)
                intensities.append(ints)
                subject_ids.append(sid)
        except Exception as e:
            print(f"[skip] {file}: {e}")
            continue

    if features:
        return {
            'features':    np.concatenate(features, axis=0),   # (total_segments, 62, 16)
            'labels':      np.concatenate(labels, axis=0),     # (total_segments,)
            'intensities': np.concatenate(intensities, axis=0),
            'subject_ids': np.concatenate(subject_ids, axis=0) # (total_segments,)
        }
    else:
        print("Warning: No valid feature files found!")
        return None


if __name__ == "__main__":
    input_dir = r"\SEED-VII\clean_data_test"
    output_dir = r"\SEED-VII\processed_features"

    print("Starting batch processing...")
    batch_process(input_dir, output_dir)

    print("\nLoading all features for analysis...")
    data = load_all_features(output_dir)

    if data is not None:
        print("\nSummary Statistics:")
        print(f"Total segments: {len(data['labels'])}")
        print(f"Feature matrix shape: {data['features'].shape}")
        print(f"Unique subjects: {np.unique(data['subject_ids'])}")
        print(f"Label distribution: {np.bincount(data['labels'][data['labels'] >= 0])}")