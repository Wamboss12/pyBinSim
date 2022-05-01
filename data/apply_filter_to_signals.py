import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
import soundfile as sf

paths_to_signals = ["../example/new_signals/Dialog_normal.wav", "../example/new_signals/music_sherlock_normal.wav",
                    "../example/new_signals/music_sherlock_alt_normal.wav"]
order = 6
lowcut = 200.0
highcut = 10000.0
N_f = 4096


for path_to_signal in paths_to_signals:
    audio, fs = sf.read(path_to_signal)
    b, a = butter(order, Wn=lowcut, fs=fs, btype='high', output="ba")
    b_2, a_2 = butter(order, Wn=[lowcut, highcut], fs=fs, btype='band', output="ba")
    audio_high_passed = lfilter(b, a, audio, axis=0)
    audio_band_passed = lfilter(b_2, a_2, audio, axis=0)
    f_bins = np.arange(N_f) * fs / N_f

    plt.plot(f_bins[:N_f//2], 20 * np.log10( np.abs(np.fft.fft(audio_high_passed[:, 1], N_f)[:N_f//2])), label="highpass")
    plt.plot(f_bins[:N_f//2], 20 * np.log10( np.abs(np.fft.fft(audio[:, 1], N_f)[:N_f//2])), label="old")
    plt.plot(f_bins[:N_f//2], 20 * np.log10( np.abs(np.fft.fft(audio_band_passed[:, 1], N_f)[:N_f//2])), label="bandpass")
    plt.xlabel('Frequency (Hz)')
    plt.xscale("log")
    plt.ylabel('Amplitude (dB)')
    plt.grid(True)
    plt.legend(loc='best')
    plt.show()

    sf.write(path_to_signal.replace("normal", f"highpassed_{int(lowcut)}"), audio_high_passed, fs, subtype="DOUBLE")
    sf.write(path_to_signal.replace("normal", f"bandpassed_{int(lowcut)}_{int(highcut)}"), audio_band_passed, fs, subtype="DOUBLE")
