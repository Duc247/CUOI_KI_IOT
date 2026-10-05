import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal

# Set font and style for scientific publication
matplotlib.rcParams['font.sans-serif'] = 'Times New Roman'
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.titlesize'] = 12
matplotlib.rcParams['axes.labelsize'] = 11
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['figure.titlesize'] = 13

base_dir = r'c:\Users\admin\Downloads\FSDD_UNet2D_DAE_VuMinhDuc-20261005T122913Z-1-001\FSDD_UNet2D_DAE_VuMinhDuc'
res_dir = os.path.join(base_dir, 'results')
art_dir = os.path.join(base_dir, 'artifacts')
ms_dir = os.path.join(base_dir, 'multiseed_3runs', 'results')
fig_dir = os.path.join(base_dir, 'figures')
os.makedirs(fig_dir, exist_ok=True)

# -------------------------------------------------------------
# Pure NumPy / SciPy Slaney Mel Filterbank & STFT
# -------------------------------------------------------------
def hz_to_mel(hz):
    hz = np.asarray(hz)
    f_min = 0.0
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(hz >= min_log_hz,
                    min_log_mel + np.log(hz / min_log_hz) / logstep,
                    (hz - f_min) / f_sp)

def mel_to_hz(mel):
    mel = np.asarray(mel)
    f_min = 0.0
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(mel >= min_log_mel,
                    min_log_hz * np.exp(logstep * (mel - min_log_mel)),
                    f_min + f_sp * mel)

def create_mel_filterbank(sr=8000, n_fft=512, n_mels=64, fmin=20, fmax=4000):
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=np.float32)
    fft_freqs = np.linspace(0, sr / 2, int(1 + n_fft // 2))
    min_mel = hz_to_mel(fmin)
    max_mel = hz_to_mel(fmax)
    mels = np.linspace(min_mel, max_mel, n_mels + 2)
    mel_f = mel_to_hz(mels)
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fft_freqs)
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))
    enorm = 2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, np.newaxis]
    return weights

W_MEL = create_mel_filterbank()

# -------------------------------------------------------------
# Hình 2.1: Phân bố thời lượng bản ghi FSDD
# -------------------------------------------------------------
print('Generating Hình 2.1...')
manifest = pd.read_csv(os.path.join(art_dir, 'manifest.csv'))
durations = manifest['duration_sec'].values

plt.figure(figsize=(6.2, 4.0), dpi=300)
plt.hist(durations, bins=35, color='#2b5c8f', edgecolor='black', alpha=0.85, rwidth=0.9)
mean_dur = np.mean(durations)
median_dur = np.median(durations)

plt.axvline(mean_dur, color='#d9534f', linestyle='--', linewidth=1.5, label=f'Trung bình: {mean_dur:.3f} s')
plt.axvline(median_dur, color='#2ca02c', linestyle=':', linewidth=1.5, label=f'Trung vị: {median_dur:.3f} s')

plt.title('Phân bố thời lượng của 3.000 bản ghi trong tập dữ liệu FSDD v1.0.10', pad=10)
plt.xlabel('Thời lượng bản ghi (giây)')
plt.ylabel('Số lượng bản ghi')
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.legend(frameon=True, edgecolor='#cccccc')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'hinh_2_1_duration_dist.png'), dpi=300)
plt.close()

# -------------------------------------------------------------
# Hình 2.2: Waveform clean và noisy minh họa
# -------------------------------------------------------------
print('Generating Hình 2.2...')
clean_wav_path = os.path.join(res_dir, 'audio_demo', 'clean.wav')
noisy_wav_path = os.path.join(res_dir, 'audio_demo', 'noisy.wav')
clean_audio, sr = sf.read(clean_wav_path)
noisy_audio, _ = sf.read(noisy_wav_path)
time_axis = np.arange(len(clean_audio)) / sr

fig, axes = plt.subplots(2, 1, figsize=(6.5, 4.2), dpi=300, sharex=True)
axes[0].plot(time_axis, clean_audio, color='#1f77b4', linewidth=0.8)
axes[0].set_title('Tín hiệu sạch (Clean Speech) – Bản ghi 0_lucas_0 (chữ số "0", người nói lucas)', fontsize=10.5)
axes[0].set_ylabel('Biên độ')
axes[0].grid(True, linestyle=':', alpha=0.6)
axes[0].set_ylim([-0.42, 0.42])

axes[1].plot(time_axis, noisy_audio, color='#d62728', linewidth=0.8)
axes[1].set_title('Tín hiệu có nhiễu (Noisy Speech) – AWGN, SNR mục tiêu 0 dB (SNR thực đo: 0,000 dB)', fontsize=10.5)
axes[1].set_xlabel('Thời gian (giây)')
axes[1].set_ylabel('Biên độ')
axes[1].grid(True, linestyle=':', alpha=0.6)
axes[1].set_ylim([-0.42, 0.42])

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'hinh_2_2_waveform_demo.png'), dpi=300)
plt.close()

# -------------------------------------------------------------
# Hình 2.3: Log-mel spectrogram clean và noisy minh họa
# -------------------------------------------------------------
print('Generating Hình 2.3...')
npz = np.load(os.path.join(art_dir, 'norm_stats.npz'))
a, b = float(npz['a']), float(npz['b'])

def wave_to_logmel_norm(y):
    if len(y) < 8000:
        y_padded = np.pad(y, (0, 8000 - len(y)), mode='constant')
    else:
        y_padded = y[:8000]
    
    f, t, Zxx = scipy.signal.stft(y_padded, fs=8000, window='hann', nperseg=512, noverlap=512-128, nfft=512, boundary='zeros', padded=True)
    power_stft = np.abs(Zxx[:, :63])**2
    mel_power = np.matmul(W_MEL, power_stft)
    log_mel = np.log(mel_power + 1e-8)
    norm_mel = (log_mel - a) / (b - a)
    norm_mel_64 = np.pad(norm_mel, ((0,0),(0,1)), mode='constant')
    return norm_mel_64

mel_clean = wave_to_logmel_norm(clean_audio)
mel_noisy = wave_to_logmel_norm(noisy_audio)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), dpi=300)
im0 = axes[0].imshow(mel_clean, origin='lower', aspect='auto', cmap='magma', vmin=0, vmax=1)
axes[0].set_title(r'Log-mel sạch chuẩn hóa (Clean $X_s$)', fontsize=11)
axes[0].set_xlabel('Chỉ số khung thời gian (Frame 0–63)')
axes[0].set_ylabel('Chỉ số dải Mel (0–63)')

im1 = axes[1].imshow(mel_noisy, origin='lower', aspect='auto', cmap='magma', vmin=0, vmax=1)
axes[1].set_title(r'Log-mel có nhiễu chuẩn hóa (Noisy $X_y$)', fontsize=11)
axes[1].set_xlabel('Chỉ số khung thời gian (Frame 0–63)')
axes[1].set_ylabel('Chỉ số dải Mel (0–63)')

cbar = fig.colorbar(im1, ax=[axes[0], axes[1]], orientation='vertical', fraction=0.03, pad=0.03)
cbar.set_label(r'Giá trị chuẩn hóa Min-Max $[0, 1]$', rotation=270, labelpad=15)

plt.savefig(os.path.join(fig_dir, 'hinh_2_3_logmel_demo.png'), dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------
# Hình 3.1: Sơ đồ tổng thể quy trình thực nghiệm
# -------------------------------------------------------------
print('Generating Hình 3.1...')
fig, ax = plt.subplots(figsize=(8.0, 5.2), dpi=300)
ax.axis('off')

boxes = [
    (0.05, 0.78, 0.22, 0.14, 'Tập FSDD Sạch\n(8 kHz, 3.000 file)\n4 Train / 1 Val / 1 Test', '#e1f5fe'),
    (0.38, 0.78, 0.24, 0.14, 'Phối trộn nhiễu có kiểm soát\nAWGN / Babble (MUSAN)\n0 dB, 5 dB, 10 dB\n+ Prefix 2.048 mẫu', '#fff9c4'),
    (0.72, 0.78, 0.23, 0.14, 'Tín hiệu quan sát y[n]\n(Noisy Waveform)', '#ffecb3'),
    (0.05, 0.44, 0.20, 0.18, 'Nhánh A: Noisy\n(Tín hiệu chưa xử lý)\nĐóng vai trò mốc chuẩn\n(Baseline không can thiệp)', '#f5f5f5'),
    (0.28, 0.44, 0.20, 0.18, 'Nhánh B: Spectral Sub.\nNoise PSD prefix (T_N=13)\nTrừ phổ (alpha=2, beta=0.01)\nPha noisy -> ISTFT', '#e8f5e9'),
    (0.51, 0.44, 0.20, 0.18, 'Nhánh C: Wiener DD\nNoise PSD prefix (T_N=13)\nƯớc lượng xi[k,t] (DD alpha=0.90)\nLọc Wiener -> ISTFT', '#ede7f6'),
    (0.74, 0.44, 0.22, 0.18, 'Nhánh D: U-Net 2D DAE\nSTFT -> Log-mel (64 dải)\nU-Net DAE (1x64x64)\nNghịch đảo NNLS -> ISTFT', '#fce4ec'),
    (0.20, 0.08, 0.60, 0.18, 'Đánh giá khách quan toàn diện (Objective Evaluation)\nSNR gain (dB) = SNR_out - SNR_in\nChất lượng âm thanh: PESQ-NB (ITU-T P.862)\nĐộ rõ tiếng nói: STOI (Short-Time Objective Intelligibility)\nChi phí xử lý: Độ trễ forward/e2e, RTF, RAM đỉnh, tham số FP32', '#e0f2f1'),
]

for x, y, w, h, t, bg in boxes:
    rect = matplotlib.patches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.015',
                                              facecolor=bg, edgecolor='#333333', linewidth=1.2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, t, ha='center', va='center', fontsize=9.2, multialignment='center')

arrow_kw = dict(arrowstyle='->', lw=1.5, color='#333333')
ax.annotate('', xy=(0.38, 0.85), xytext=(0.27, 0.85), arrowprops=arrow_kw)
ax.annotate('', xy=(0.72, 0.85), xytext=(0.62, 0.85), arrowprops=arrow_kw)

ax.annotate('', xy=(0.15, 0.62), xytext=(0.75, 0.78), arrowprops=dict(arrowstyle='->', lw=1.2, color='#555555', connectionstyle='angle,angleA=0,angleB=90,rad=5'))
ax.annotate('', xy=(0.38, 0.62), xytext=(0.80, 0.78), arrowprops=dict(arrowstyle='->', lw=1.2, color='#555555', connectionstyle='angle,angleA=0,angleB=90,rad=5'))
ax.annotate('', xy=(0.61, 0.62), xytext=(0.83, 0.78), arrowprops=dict(arrowstyle='->', lw=1.2, color='#555555', connectionstyle='angle,angleA=0,angleB=90,rad=5'))
ax.annotate('', xy=(0.85, 0.62), xytext=(0.85, 0.78), arrowprops=arrow_kw)

for bx in [0.15, 0.38, 0.61, 0.85]:
    ax.annotate('', xy=(0.50, 0.26), xytext=(bx, 0.44), arrowprops=dict(arrowstyle='->', lw=1.2, color='#555555', connectionstyle='arc3,rad=0.05'))

plt.savefig(os.path.join(fig_dir, 'hinh_3_1_pipeline_diagram.png'), dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------
# Hình 3.2: Kiến trúc U-Net 2D DAE
# -------------------------------------------------------------
print('Generating Hình 3.2...')
fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=300)
ax.axis('off')

blocks = [
    (0.08, 0.72, 0.16, 0.14, 'Input\n1x64x64\nLog-mel chuẩn hóa', '#e0e0e0'),
    (0.08, 0.52, 0.16, 0.14, 'Encoder 1\nDoubleConv\n16x64x64', '#bbdefb'),
    (0.18, 0.36, 0.16, 0.14, 'Encoder 2\nDoubleConv\n32x32x32', '#90caf9'),
    (0.28, 0.20, 0.16, 0.14, 'Encoder 3\nDoubleConv\n64x16x16', '#64b5f6'),
    (0.42, 0.04, 0.16, 0.14, 'Bottleneck\nDoubleConv\n128x8x8', '#42a5f5'),
    (0.56, 0.20, 0.16, 0.14, 'Decoder 3\nUp + Cat + DConv\n64x16x16', '#ffcc80'),
    (0.66, 0.36, 0.16, 0.14, 'Decoder 2\nUp + Cat + DConv\n32x32x32', '#ffa726'),
    (0.76, 0.52, 0.16, 0.14, 'Decoder 1\nUp + Cat + DConv\n16x64x64', '#ff9800'),
    (0.76, 0.72, 0.16, 0.14, 'Output\nConv2d 1x1 Linear\n1x64x64', '#c8e6c9'),
]

for x, y, w, h, t, bg in blocks:
    rect = matplotlib.patches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012',
                                              facecolor=bg, edgecolor='#333333', linewidth=1.1)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, t, ha='center', va='center', fontsize=8.8, multialignment='center')

ax.annotate('MaxPool 2x2', xy=(0.20, 0.50), xytext=(0.14, 0.52), arrowprops=dict(arrowstyle='->', lw=1.2, color='#1565c0'), fontsize=8, color='#1565c0')
ax.annotate('MaxPool 2x2', xy=(0.30, 0.34), xytext=(0.24, 0.36), arrowprops=dict(arrowstyle='->', lw=1.2, color='#1565c0'), fontsize=8, color='#1565c0')
ax.annotate('MaxPool 2x2', xy=(0.44, 0.18), xytext=(0.34, 0.20), arrowprops=dict(arrowstyle='->', lw=1.2, color='#1565c0'), fontsize=8, color='#1565c0')

ax.annotate('Up 2x', xy=(0.62, 0.20), xytext=(0.56, 0.14), arrowprops=dict(arrowstyle='->', lw=1.2, color='#e65100'), fontsize=8, color='#e65100')
ax.annotate('Up 2x', xy=(0.72, 0.36), xytext=(0.66, 0.30), arrowprops=dict(arrowstyle='->', lw=1.2, color='#e65100'), fontsize=8, color='#e65100')
ax.annotate('Up 2x', xy=(0.82, 0.52), xytext=(0.76, 0.46), arrowprops=dict(arrowstyle='->', lw=1.2, color='#e65100'), fontsize=8, color='#e65100')
ax.annotate('', xy=(0.84, 0.72), xytext=(0.84, 0.66), arrowprops=dict(arrowstyle='->', lw=1.2, color='#2e7d32'))

skip_props = dict(arrowstyle='->', lw=1.5, linestyle='--', color='#d32f2f')
ax.annotate('', xy=(0.76, 0.59), xytext=(0.24, 0.59), arrowprops=skip_props)
ax.text(0.50, 0.605, 'Skip Connection 1 (16x64x64)', ha='center', va='bottom', fontsize=8.2, color='#d32f2f', weight='bold')

ax.annotate('', xy=(0.66, 0.43), xytext=(0.34, 0.43), arrowprops=skip_props)
ax.text(0.50, 0.445, 'Skip Connection 2 (32x32x32)', ha='center', va='bottom', fontsize=8.2, color='#d32f2f', weight='bold')

ax.annotate('', xy=(0.56, 0.27), xytext=(0.44, 0.27), arrowprops=skip_props)
ax.text(0.50, 0.285, 'Skip Connection 3 (64x16x16)', ha='center', va='bottom', fontsize=8.2, color='#d32f2f', weight='bold')

ax.text(0.50, 0.94, 'Kiến trúc U-Net 2D DAE: 449.489 tham số | Dung lượng FP32: 1,80 MB | Đầu vào/Đầu ra: 1x64x64',
        ha='center', va='center', fontsize=9.5, weight='bold', bbox=dict(boxstyle='square,pad=0.4', facecolor='#f5f5f5', edgecolor='#999999'))

plt.savefig(os.path.join(fig_dir, 'hinh_3_2_unet_architecture.png'), dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------
# Hình 5.1: Đường cong huấn luyện U-Net chính
# -------------------------------------------------------------
print('Generating Hình 5.1...')
main_hist = pd.read_csv(os.path.join(res_dir, 'unet_main_history.csv'))
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), dpi=300)

axes[0].plot(main_hist['epoch'], main_hist['train_loss'], marker='o', markersize=4, color='#1f77b4', label='Train Loss')
axes[0].plot(main_hist['epoch'], main_hist['val_loss'], marker='s', markersize=4, color='#d62728', label='Validation Loss')
axes[0].axvline(5, color='#2ca02c', linestyle='--', label='Best Val Epoch (5)')
axes[0].set_title(r'Hàm mất mát tổng hợp (Hybrid Loss $L_1 + L_2$)', fontsize=10.5)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (Log-mel)')
axes[0].grid(True, linestyle=':', alpha=0.6)
axes[0].legend(frameon=True, edgecolor='#cccccc')

axes[1].plot(main_hist['epoch'], main_hist['val_l1'], marker='^', markersize=4, color='#9467bd', label='Val L1 Loss')
axes[1].plot(main_hist['epoch'], main_hist['val_l2'], marker='v', markersize=4, color='#8c564b', label='Val L2 Loss')
axes[1].set_title('Phân rã thành phần L1 và L2 trên Validation', fontsize=10.5)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Giá trị mất mát')
axes[1].grid(True, linestyle=':', alpha=0.6)
axes[1].legend(frameon=True, edgecolor='#cccccc')

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'hinh_5_1_training_curves.png'), dpi=300)
plt.close()

# -------------------------------------------------------------
# Hình 5.2: Đường cong huấn luyện U-Net bỏ skip
# -------------------------------------------------------------
print('Generating Hình 5.2...')
skip_hist = pd.read_csv(os.path.join(res_dir, 'unet_skip_off_history.csv'))
plt.figure(figsize=(6.2, 3.8), dpi=300)
plt.plot(skip_hist['epoch'], skip_hist['train_loss'], marker='o', markersize=4, color='#1f77b4', label='Train Loss (Skip-Off)')
plt.plot(skip_hist['epoch'], skip_hist['val_loss'], marker='s', markersize=4, color='#d62728', label='Validation Loss (Skip-Off)')
plt.axvline(6, color='#2ca02c', linestyle='--', label='Best Val Epoch (6, loss=0,0414)')
plt.title('Đường cong huấn luyện U-Net 2D DAE phiên bản bỏ Skip Connection', pad=10)
plt.xlabel('Epoch')
plt.ylabel(r'Hybrid Loss ($L_1 + L_2$)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(frameon=True, edgecolor='#cccccc')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'hinh_5_2_skipoff_training_curves.png'), dpi=300)
plt.close()

# -------------------------------------------------------------
# Hình 5.3: SNR gain theo 6 điều kiện và phương pháp
# -------------------------------------------------------------
print('Generating Hình 5.3...')
sum_cond = pd.read_csv(os.path.join(res_dir, 'summary_by_condition.csv'))

conditions = [
    ('awgn', 0, 'AWGN 0 dB'),
    ('awgn', 5, 'AWGN 5 dB'),
    ('awgn', 10, 'AWGN 10 dB'),
    ('babble', 0, 'Babble 0 dB'),
    ('babble', 5, 'Babble 5 dB'),
    ('babble', 10, 'Babble 10 dB'),
]
cond_labels = [c[2] for c in conditions]

methods = ['Noisy', 'SpectralSubtraction', 'Wiener', 'UNet2D_DAE']
method_display = ['Noisy (0 dB)', 'Spectral Sub.', 'Wiener DD', 'U-Net 2D DAE']
colors = ['#7f7f7f', '#2ca02c', '#9467bd', '#1f77b4']

x = np.arange(len(cond_labels))
width = 0.20

plt.figure(figsize=(7.5, 4.2), dpi=300)
for idx, (m, m_name, col) in enumerate(zip(methods, method_display, colors)):
    vals = []
    for n_type, snr_t, _ in conditions:
        row = sum_cond[(sum_cond['method']==m) & (sum_cond['noise_type']==n_type) & (sum_cond['snr_target']==snr_t)]
        vals.append(float(row['snr_gain_mean'].values[0]))
    plt.bar(x + (idx - 1.5)*width, vals, width, label=m_name, color=col, edgecolor='black', linewidth=0.6)

plt.axhline(0, color='black', linewidth=0.8, linestyle='-')
plt.xticks(x, cond_labels)
plt.ylabel('SNR Gain (dB)')
plt.title('Mức cải thiện SNR (SNR gain) của các phương pháp trên sáu điều kiện kiểm thử', pad=10)
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.legend(frameon=True, edgecolor='#cccccc')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'hinh_5_3_snr_gain.png'), dpi=300)
plt.close()

# -------------------------------------------------------------
# Hình 5.4: Delta PESQ theo 6 điều kiện và phương pháp
# -------------------------------------------------------------
print('Generating Hình 5.4...')
plt.figure(figsize=(7.5, 4.2), dpi=300)
for idx, (m, m_name, col) in enumerate(zip(methods[1:], method_display[1:], colors[1:])):
    vals = []
    for n_type, snr_t, _ in conditions:
        row = sum_cond[(sum_cond['method']==m) & (sum_cond['noise_type']==n_type) & (sum_cond['snr_target']==snr_t)]
        vals.append(float(row['delta_pesq_mean'].values[0]))
    plt.bar(x + (idx - 1)*width*1.2, vals, width*1.2, label=m_name, color=col, edgecolor='black', linewidth=0.6)

plt.axhline(0, color='black', linewidth=0.8, linestyle='-')
plt.xticks(x, cond_labels)
plt.ylabel(r'$\Delta$PESQ (Điểm chất lượng cải thiện)')
plt.title(r'Mức cải thiện chất lượng cảm nhận ($\Delta$PESQ) trên sáu điều kiện kiểm thử', pad=10)
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.legend(frameon=True, edgecolor='#cccccc')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'hinh_5_4_delta_pesq.png'), dpi=300)
plt.close()

# -------------------------------------------------------------
# Hình 5.5: Delta STOI theo 6 điều kiện và phương pháp
# -------------------------------------------------------------
print('Generating Hình 5.5...')
plt.figure(figsize=(7.5, 4.2), dpi=300)
for idx, (m, m_name, col) in enumerate(zip(methods[1:], method_display[1:], colors[1:])):
    vals = []
    for n_type, snr_t, _ in conditions:
        row = sum_cond[(sum_cond['method']==m) & (sum_cond['noise_type']==n_type) & (sum_cond['snr_target']==snr_t)]
        vals.append(float(row['delta_stoi_mean'].values[0]))
    plt.bar(x + (idx - 1)*width*1.2, vals, width*1.2, label=m_name, color=col, edgecolor='black', linewidth=0.6)

plt.axhline(0, color='black', linewidth=0.8, linestyle='-')
plt.xticks(x, cond_labels)
plt.ylabel(r'$\Delta$STOI (Độ rõ tiếng nói cải thiện)')
plt.title(r'Mức cải thiện độ rõ tiếng nói ($\Delta$STOI) trên sáu điều kiện kiểm thử', pad=10)
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.legend(frameon=True, edgecolor='#cccccc')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'hinh_5_5_delta_stoi.png'), dpi=300)
plt.close()

# -------------------------------------------------------------
# Hình 5.6: Ablation Skip: Main vs Skip-Off
# -------------------------------------------------------------
print('Generating Hình 5.6...')
abl_df = pd.read_csv(os.path.join(res_dir, 'ablation_skip_summary.csv'))
fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.8), dpi=300)

metrics = [('snr_gain_mean', 'SNR Gain (dB)', axes[0]),
           ('delta_pesq_mean', r'$\Delta$PESQ', axes[1]),
           ('delta_stoi_mean', r'$\Delta$STOI', axes[2])]

x_abl = np.arange(len(cond_labels))
w_abl = 0.35

for col_name, label_name, ax in metrics:
    vals_main = []
    vals_skip = []
    for n_type, snr_t, _ in conditions:
        r_m = abl_df[(abl_df['method']=='UNet2D_DAE') & (abl_df['noise_type']==n_type) & (abl_df['snr_target']==snr_t)]
        r_s = abl_df[(abl_df['method']=='UNet2D_DAE_SkipOff') & (abl_df['noise_type']==n_type) & (abl_df['snr_target']==snr_t)]
        vals_main.append(float(r_m[col_name].values[0]))
        vals_skip.append(float(r_s[col_name].values[0]))
        
    ax.bar(x_abl - w_abl/2, vals_main, w_abl, label='U-Net Chính (Có Skip)', color='#1f77b4', edgecolor='black', linewidth=0.6)
    ax.bar(x_abl + w_abl/2, vals_skip, w_abl, label='U-Net Bỏ Skip (Skip-Off)', color='#ff7f0e', edgecolor='black', linewidth=0.6)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='-')
    ax.set_xticks(x_abl)
    ax.set_xticklabels(['A0', 'A5', 'A10', 'B0', 'B5', 'B10'], fontsize=8.5)
    ax.set_ylabel(label_name)
    ax.set_title(label_name, fontsize=11)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    if col_name == 'snr_gain_mean':
        ax.legend(fontsize=8, frameon=True, edgecolor='#cccccc')

fig.suptitle('Phân tích bóc tách (Ablation): Ảnh hưởng của Skip Connection đối với hiệu năng mô hình', y=1.02, fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'hinh_5_6_ablation_skip.png'), dpi=300)
plt.close()

# -------------------------------------------------------------
# Hình 5.7: Biến thiên giữa các seed (Multi-seed variation)
# -------------------------------------------------------------
print('Generating Hình 5.7...')
ms_cond = pd.read_csv(os.path.join(ms_dir, 'mean_std_3seeds_by_condition.csv'))
fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.8), dpi=300)

ms_metrics = [
    ('snr_gain_mean_across_seeds_mean', 'snr_gain_mean_across_seeds_std', 'SNR Gain (dB)', axes[0]),
    ('delta_pesq_mean_across_seeds_mean', 'delta_pesq_mean_across_seeds_std', r'$\Delta$PESQ', axes[1]),
    ('delta_stoi_mean_across_seeds_mean', 'delta_stoi_mean_across_seeds_std', r'$\Delta$STOI', axes[2]),
]

for mean_col, std_col, label_name, ax in ms_metrics:
    m_vals = []
    s_vals = []
    for n_type, snr_t, _ in conditions:
        r = ms_cond[(ms_cond['method']=='UNet2D_DAE') & (ms_cond['noise_type']==n_type) & (ms_cond['snr_target']==snr_t)]
        m_vals.append(float(r[mean_col].values[0]))
        s_vals.append(float(r[std_col].values[0]))
        
    ax.errorbar(x_abl, m_vals, yerr=s_vals, fmt='o-', capsize=4, color='#1f77b4', ecolor='#d62728', elinewidth=1.5, markeredgecolor='black', label='U-Net Mean ± Std')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='-')
    ax.set_xticks(x_abl)
    ax.set_xticklabels(['A0', 'A5', 'A10', 'B0', 'B5', 'B10'], fontsize=8.5)
    ax.set_ylabel(label_name)
    ax.set_title(label_name, fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.6)

fig.suptitle('Độ ổn định huấn luyện qua 3 Random Seeds (42, 123, 999) trên sáu điều kiện kiểm thử', y=1.02, fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'hinh_5_7_multiseed_variation.png'), dpi=300)
plt.close()

# -------------------------------------------------------------
# Hình 6.1: So sánh thời gian xử lý Forward vs End-to-End
# -------------------------------------------------------------
print('Generating Hình 6.1...')
bench = pd.read_csv(os.path.join(res_dir, 'benchmark_full.csv')).iloc[0]

fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=300)
categories = ['Forward Mạng (Network Only)', 'Toàn Chuỗi (End-to-End Pipeline)']
means = [bench['forward_mean_ms'], bench['e2e_mean_ms']]
medians = [bench['forward_median_ms'], bench['e2e_median_ms']]
p95s = [bench['forward_p95_ms'], bench['e2e_p95_ms']]

x_b = np.arange(len(categories))
w_b = 0.25

rects1 = ax.bar(x_b - w_b, means, w_b, label='Trung bình (Mean)', color='#1f77b4', edgecolor='black', linewidth=0.6)
rects2 = ax.bar(x_b, medians, w_b, label='Trung vị (Median)', color='#2ca02c', edgecolor='black', linewidth=0.6)
rects3 = ax.bar(x_b + w_b, p95s, w_b, label='Phân vị 95 (P95)', color='#d62728', edgecolor='black', linewidth=0.6)

ax.set_ylabel('Thời gian thực thi (ms / block 1 giây)')
ax.set_title('So sánh độ trễ xử lý suy luận của U-Net 2D DAE trên CPU thực tế', pad=10)
ax.set_xticks(x_b)
ax.set_xticklabels(categories, fontsize=10.5)
ax.grid(axis='y', linestyle=':', alpha=0.6)
ax.legend(frameon=True, edgecolor='#cccccc')

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f} ms',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.5)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'hinh_6_1_latency_breakdown.png'), dpi=300)
plt.close()

print('ALL 13 FIGURES GENERATED SUCCESSFULLY!')
