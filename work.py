import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import scipy.io
import scipy.optimize

N_f = 8192
path_to_rirs = "example/RIRs_ListeningLab_frontLS_125-325cm_3repetitions_48kHz.mat"
brir = "example/brirs/BRIR_Front_Audiolab/125/brir0.wav"
path_to_hrir = "example/HRIR_KEMAR_ED.mat"
path_to_left_tweeter = "example/leftTweeter.mat"
path_to_driver_database = "example/DriverResponse_KEMAR_AC0010_1.mat"

left_tweeter_db = scipy.io.loadmat(path_to_driver_database).get("M_data")
left_tweeter_db = left_tweeter_db[:, 0, 0]
left_tweeter_db /= np.max(left_tweeter_db)
spectrum_tweeter_db = 20*np.log10( np.abs(np.fft.fft(left_tweeter_db, N_f)) )

left_tweeters = scipy.io.loadmat(path_to_left_tweeter).get("h_ear")
left_tweeter = left_tweeters[:, 0]
left_tweeter /= np.max(left_tweeter)
spectrum_tweeter = 20*np.log10( np.abs(np.fft.fft(left_tweeter, N_f)) )

hrirs = scipy.io.loadmat(path_to_hrir).get("M_data")
hrir = hrirs[:, 0, 0]
spectrum_hrir = 20*np.log10( np.abs(np.fft.fft(hrir, N_f)) ) - 10
freq_bin2 = 44100/N_f
freq_bins2 = np.arange(start=0, stop=4096, step=1) * freq_bin2

rirs = scipy.io.loadmat(path_to_rirs).get("rir_front_LL")
# samples x repition x position
rir_front = rirs[:, 0, 0]
spectrum_rir = 20*np.log10( np.abs(np.fft.fft(rir_front, N_f)) )

audio, sf = sf.read(brir)
freq_bin = sf/N_f
freq_bins = np.arange(start=0, stop=4096, step=1) * freq_bin
channel_1 = audio[:, 0]
spectrum = 20*np.log10( np.abs(np.fft.fft(channel_1, N_f)) )

plt.plot(freq_bins[:N_f//2], spectrum[:N_f//2], label="BRIR")
plt.plot(freq_bins[:N_f//2], spectrum_rir[:N_f//2], label="RIR")
plt.plot(freq_bins2[:N_f//2], spectrum_hrir[:N_f//2], label="HRIR")
#plt.plot(freq_bins[:N_f//2], spectrum_tweeter[:N_f//2], label="Tweeter IDMT")
#plt.plot(freq_bins2[:N_f//2], spectrum_tweeter_db[:N_f//2], label="Tweeter Database")
plt.grid()
plt.legend()
plt.xscale("log")
plt.show()
print()