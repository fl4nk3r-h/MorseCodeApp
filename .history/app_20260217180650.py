import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import numpy as np
import pyaudio  # For encoder
import wave  # For encoder and decoder output
import math
import threading
import os
import scipy.io.wavfile as scipy_wav  # For decoder
import scipy.signal as scipy_signal  # For decoder
from sklearn.cluster import KMeans  # For decoder
import sys  

#author: fl

# --- Morse Code Dictionaries (Shared) ---
MORSE_CODE_DICT_ENCODE = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.', 'G': '--.', 'H': '....',
    'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---', 'P': '.--.',
    'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',
    'Y': '-.--', 'Z': '--..', '1': '.----', '2': '..---', '3': '...--', '4': '....-', '5': '.....',
    '6': '-....', '7': '--...', '8': '---..', '9': '----.', '0': '-----',
    ',': '--..--', '.': '.-.-.-', '?': '..--..', '/': '-..-.', '-': '-....-', '(': '-.--.', ')': '-.--.-',
    ' ': ' ',
    '!': '-.-.--', '@': '.--.-.', '&': '.-...', '=': '-...-', '+': '.-.-.', ':': '---...',
    ';': '-.-.-.', '_': '..--.-', '$': '...-..-', '"': '.-..-.'
}

MORSE_CODE_DICT_DECODE = {
    # Letters
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E',
    '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J',
    '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O',
    '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
    '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y',
    '--..': 'Z',

    # Numbers
    '-----': '0', '.----': '1', '..---': '2', '...--': '3', '....-': '4',
    '.....': '5', '-....': '6', '--...': '7', '---..': '8', '----.': '9',

    # Punctuation
    '.-.-.-': '.', '--..--': ',', '..--..': '?', '.----.': "'",
    '-.-.--': '!', '-..-.': '/', '-.--.': '(', '-.--.-': ')',
    '.-...': '&', '---...': ':', '-.-.-.': ';', '-...-': '=',
    '.-.-.': '+', '-....-': '-', '..--.-': '_', '.-..-.': '"',
    '...-..-': '$', '.--.-.': '@',

    # Prosigns (optional, control signals)
    '...-.-': 'SK',  # End of contact
    '.-.-.': 'AR',   # End of message
    '-...-.-': 'BK', # Break
    '-.-': 'KN',     # Go ahead only

    # Space representation (for decoder logic)
    '': ' '  # Useful when splitting Morse code words
}


# --- Global Helper Functions (Stateless utilities) ---
def cluster_durations_global(durations, n_clusters):
    durations_arr = np.array(durations).reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(durations_arr)
    centers = sorted(kmeans.cluster_centers_.flatten())
    return kmeans.labels_, centers


# --- Main Application Class ---
class MorseCodeApp:
    def __init__(self, master):
        self.master = master
        master.title("Morse Code Toolkit Pro")
        master.minsize(500, 600)

        self.pyaudio_instance = pyaudio.PyAudio()
        self.encoder_stream = None
        self.encoder_is_playing = False
        self.encoder_playback_thread = None
        self.decoder_thread = None

        self.style = ttk.Style()
        try:
            self.style.theme_use('clam')
        except tk.TclError:
            self.style.theme_use('default')

        self.style.configure('TLabel', font=('Segoe UI', 10))
        self.style.configure('TButton', font=('Segoe UI', 10, 'bold'), padding=(10, 5))
        self.style.configure('TLabelframe.Label', font=('Segoe UI', 11, 'bold'), padding=(0, 0, 0, 5))
        self.style.configure('TLabelframe', padding=10)
        self.style.configure('Parameter.TLabel', font=('Segoe UI', 9))
        self.style.configure('Status.TLabel', font=('Segoe UI', 9), padding=3)
        self.style.configure('Accent.TButton', font=('Segoe UI', 11, 'bold'), foreground='white')
        self.style.map('Accent.TButton', background=[('active', '#45a049'), ('pressed', '#3e8e41')])

        self.notebook = ttk.Notebook(master, padding="5 5 5 5")
        self.encoder_tab = ttk.Frame(self.notebook, padding="10 10 10 10")
        self.decoder_tab = ttk.Frame(self.notebook, padding="10 10 10 10")
        self.notebook.add(self.encoder_tab, text=' Morse Encoder / Generator ')
        self.notebook.add(self.decoder_tab, text=' Morse Decoder (from WAV) ')
        self.notebook.pack(expand=True, fill='both')

        self.encoder_params = {}
        self.build_encoder_tab_ui()
        self.decoder_params = {}
        self.decoder_results = {}
        self.build_decoder_tab_ui()

        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(master, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W,
                               style='Status.TLabel')
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=2, pady=2)
        master.protocol("WM_DELETE_WINDOW", self.on_closing)

    # --- Core DSP/Helper methods for the class (used by decoder) ---
    def _spectral_noise_reduction_logic(self, data, rate, n_fft=2048, hop_length=512,
                                        noise_clip_percentile=25,
                                        gain_type='wiener_like_magnitude',
                                        over_subtraction_factor=1.0,
                                        power_exponent=1.0):
        if len(data) < n_fft:
            # print("[WARNING SNR Method] Data too short for spectral noise reduction.")
            return data
        window = 'hann'
        f, t, Zxx = scipy_signal.stft(data, fs=rate, window=window, nperseg=n_fft, noverlap=n_fft - hop_length)
        Sxx = np.abs(Zxx)
        if gain_type == 'wiener_like_power':
            Sxx_for_noise_est = Sxx ** 2
            noise_profile_est = np.percentile(Sxx_for_noise_est, noise_clip_percentile, axis=1)
            noise_profile_est = np.maximum(noise_profile_est, 1e-12)
        else:
            Sxx_for_noise_est = Sxx
            noise_profile_est = np.percentile(Sxx_for_noise_est, noise_clip_percentile, axis=1)
            noise_profile_est = np.maximum(noise_profile_est, 1e-7)
        if gain_type == 'wiener_like_power':
            gain = np.maximum(0,
                              1 - (over_subtraction_factor * noise_profile_est[:, np.newaxis] / (
                                          Sxx ** 2 + 1e-9))) ** 0.5
        elif gain_type == 'wiener_like_magnitude':
            signal_mag = Sxx + 1e-9
            noise_mag_est = over_subtraction_factor * noise_profile_est[:, np.newaxis]
            gain = np.maximum(0, 1 - (noise_mag_est / signal_mag) ** power_exponent)
        elif gain_type == 'simple_gate':
            threshold_val = noise_profile_est[:, np.newaxis] * over_subtraction_factor
            gain = np.where(Sxx >= threshold_val, 1.0, 0.1)
        else:
            # print(f"[WARNING SNR Method] Unknown gain_type: {gain_type}. Skipping SNR.")
            return data
        Zxx_denoised = Zxx * gain
        _, data_denoised = scipy_signal.istft(Zxx_denoised, fs=rate, window=window, nperseg=n_fft,
                                              noverlap=n_fft - hop_length)
        if len(data_denoised) > len(data):
            data_denoised = data_denoised[:len(data)]
        elif len(data_denoised) < len(data):
            data_denoised = np.pad(data_denoised, (0, len(data) - len(data_denoised)), 'constant')
        return data_denoised.astype(data.dtype)

    def _bandpass_filter_logic(self, data, rate, lowcut, highcut, order=4):
        nyq = 0.5 * rate
        low = max(lowcut / nyq, 1e-5)
        high = min(highcut / nyq, 0.999)
        if low >= high: raise ValueError(
            f"BPF Method: Invalid normalized freqs: low={low}, high={high}. Orig: {lowcut}, {highcut}")
        b, a = scipy_signal.butter(order, [low, high], btype='band')
        return scipy_signal.filtfilt(b, a, data)

    def _detect_dominant_frequency_logic(self, data, rate, min_freq=200, max_freq=1500,
                                         typical_morse_range=(400, 1000)):
        if len(data) < 2: return None
        data_no_dc = data - np.mean(data)
        if np.all(np.abs(data_no_dc) < 1e-9): return None
        fft_values = np.fft.rfft(data_no_dc)
        fft_freqs = np.fft.rfftfreq(len(data_no_dc), d=1 / rate)
        fft_magnitude = np.abs(fft_values)
        if len(fft_magnitude) == 0: return None
        valid_indices = np.where((fft_freqs >= min_freq) & (fft_freqs <= max_freq))[0]
        if len(valid_indices) == 0:
            # print(f"[Warning DDF Method] No FFT energy in primary range [{min_freq}-{max_freq}Hz]. Widening search.")
            valid_indices = np.where(fft_freqs >= 50)[0]
            if len(valid_indices) == 0: return None
        freqs_subset = fft_freqs[valid_indices]
        mags_subset = fft_magnitude[valid_indices]
        if len(mags_subset) == 0 or np.max(mags_subset) < 1e-5: return None
        preferred_indices = \
        np.where((freqs_subset >= typical_morse_range[0]) & (freqs_subset <= typical_morse_range[1]))[0]
        best_freq = None
        if len(preferred_indices) > 0:
            dominant_idx_preferred = preferred_indices[np.argmax(mags_subset[preferred_indices])]
            best_freq = freqs_subset[dominant_idx_preferred]
        else:
            dominant_idx_overall = np.argmax(mags_subset)
            best_freq = freqs_subset[dominant_idx_overall]
            # print(f"[Info DDF Method] No peak in preferred Morse range. Using best from wider search.")
        return abs(best_freq) if best_freq is not None else None

    def add_slider(self, parent, param_dict, param_name, label_text, min_val, max_val, initial_val, resolution=1,
                   is_int=False, width=20):
        container = ttk.Frame(parent)
        container.pack(fill="x", padx=5, pady=(4, 3))
        ttk.Label(container, text=label_text, width=width).grid(row=0, column=0, sticky="w", padx=(0, 5))
        var_type = tk.IntVar if is_int else tk.DoubleVar
        param_dict[param_name] = var_type(value=initial_val)
        scale = ttk.Scale(container, from_=min_val, to=max_val, orient="horizontal", variable=param_dict[param_name],
                          command=lambda val, p=param_name, pd=param_dict: self._update_slider_label(val, p, pd))
        scale.grid(row=0, column=1, sticky="ew", padx=5)
        container.columnconfigure(1, weight=1)
        param_dict[param_name + '_label'] = tk.StringVar()
        value_label = ttk.Label(container, textvariable=param_dict[param_name + '_label'], width=7,
                                style='Parameter.TLabel', anchor='e')
        value_label.grid(row=0, column=2, sticky="e", padx=(5, 0))
        self._update_slider_label(str(initial_val), param_name, param_dict)
        if is_int: param_dict[param_name].set(int(initial_val))

    def add_combobox(self, parent, param_dict, param_name, label_text, discrete_values, initial_val, width=20):
        container = ttk.Frame(parent)
        container.pack(fill="x", padx=5, pady=(4, 5))
        ttk.Label(container, text=label_text, width=width).grid(row=0, column=0, sticky="w", padx=(0, 5))
        s_initial_val = str(initial_val)
        s_discrete_values = [str(v) for v in discrete_values]
        if s_initial_val not in s_discrete_values: s_initial_val = s_discrete_values[0]
        param_dict[param_name] = tk.StringVar(value=s_initial_val)
        combo = ttk.Combobox(container, textvariable=param_dict[param_name], values=s_discrete_values,
                             state="readonly", width=12, font=('Segoe UI', 9))
        combo.grid(row=0, column=1, sticky="w", padx=5)

    def _update_slider_label(self, val_str, param_name, param_dict):
        try:
            val = float(val_str)
            var = param_dict[param_name]
            label_var = param_dict[param_name + '_label']
            is_int_param = isinstance(var, tk.IntVar)
            if is_int_param:
                label_var.set(str(int(val)))
            else:
                if 'volume' in param_name:
                    label_var.set(f"{val:.2f}")
                elif 'ratio' in param_name or 'space_dots' in param_name or 'nr_over_subtraction_factor' in param_name or 'threshold_std_factor' in param_name:
                    label_var.set(f"{val:.1f}")
                else:
                    label_var.set(str(int(val)))
        except (ValueError, KeyError, AttributeError):
            pass

    def get_param_values(self, param_dict):
        values = {}
        for key, var in param_dict.items():
            if not key.endswith('_label'):
                try:
                    val_str = var.get()
                    if isinstance(var, tk.IntVar):
                        values[key] = int(val_str)
                    elif isinstance(var, tk.DoubleVar):
                        values[key] = float(val_str)
                    elif isinstance(var, tk.BooleanVar):
                        values[key] = var.get()  # BooleanVar.get() returns bool
                    else:
                        try:
                            values[key] = int(val_str)
                        except ValueError:
                            try:
                                values[key] = float(val_str)
                            except ValueError:
                                values[key] = val_str
                except tk.TclError:
                    if 'sample_rate_hz' in key:
                        values[key] = 44100
                    elif 'char_wpm' in key:
                        values[key] = 20
                    elif 'frequency_hz' in key:
                        values[key] = 700
                    elif 'nr_gain_type' in key:
                        values[key] = 'wiener_like_magnitude'
                    elif isinstance(param_dict.get(key), tk.BooleanVar):
                        values[key] = True
                    else:
                        values[key] = 0
        return values

    def build_encoder_tab_ui(self):
        parent = self.encoder_tab
        text_frame = ttk.LabelFrame(parent, text="Message to Encode")
        text_frame.pack(padx=5, pady=10, fill="x")
        self.encoder_params['text_to_encode'] = tk.StringVar(value="Jai Hind")
        ttk.Entry(text_frame, textvariable=self.encoder_params['text_to_encode'], font=('Segoe UI', 11)).pack(padx=10,
                                                                                                              pady=10,
                                                                                                              fill="x",
                                                                                                              expand=True)

        all_params_frame = ttk.Frame(parent)
        all_params_frame.pack(padx=5, pady=5, fill="x", expand=True)
        core_params_frame = ttk.LabelFrame(all_params_frame, text="Tone & Speed")
        core_params_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.add_slider(core_params_frame, self.encoder_params, 'char_wpm', "Character WPM:", 5, 60, 20)
        self.add_slider(core_params_frame, self.encoder_params, 'frequency_hz', "Frequency (Hz):", 200, 10000, 700)
        self.add_slider(core_params_frame, self.encoder_params, 'volume', "Volume:", 0.0, 1.0, 0.67, resolution=0.01)
        self.add_slider(core_params_frame, self.encoder_params, 'ramp_ms', "Ramp (ms):", 0, 50, 5)

        timing_params_frame = ttk.LabelFrame(all_params_frame, text="Element Timing")
        timing_params_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.add_slider(timing_params_frame, self.encoder_params, 'dash_dot_ratio', "Dash/Dot Ratio:", 2.0, 4.0, 3.0,
                        resolution=0.1)
        self.add_slider(timing_params_frame, self.encoder_params, 'inter_char_space_dots', "Char Space (dots):", 1, 10,
                        3, resolution=0.5)
        self.add_slider(timing_params_frame, self.encoder_params, 'word_space_dots', "Word Space (dots):", 3, 20, 7,
                        resolution=0.5)
        all_params_frame.columnconfigure(0, weight=1);
        all_params_frame.columnconfigure(1, weight=1)

        audio_settings_frame = ttk.LabelFrame(parent, text="Audio Settings")
        audio_settings_frame.pack(padx=5, pady=10, fill="x")
        self.add_combobox(audio_settings_frame, self.encoder_params, 'sample_rate_hz', "Sample Rate (Hz):",
                          [8000, 16000, 22050, 44100, 48000], 44100)

        action_frame = ttk.Frame(parent, padding="10 0 0 0")
        action_frame.pack(padx=5, pady=15, fill="x")
        self.encoder_play_button = ttk.Button(action_frame, text="▶ Play", command=self.encoder_toggle_play_stop,
                                              style='Accent.TButton')
        self.encoder_play_button.pack(side="left", padx=10, expand=True, fill="x", ipady=5)
        self.encoder_save_button = ttk.Button(action_frame, text="💾 Save WAV", command=self.encoder_save_wav)
        self.encoder_save_button.pack(side="left", padx=10, expand=True, fill="x", ipady=5)

    def generate_morse_audio_data(self):
        p_vals = self.get_param_values(self.encoder_params)
        text = p_vals.get('text_to_encode', "").upper();
        char_wpm = p_vals.get('char_wpm', 20);
        frequency = p_vals.get('frequency_hz', 700)
        volume = p_vals.get('volume', 0.8);
        sample_rate = int(p_vals.get('sample_rate_hz', 44100));
        dash_dot_ratio = p_vals.get('dash_dot_ratio', 3.0)
        inter_char_dots = p_vals.get('inter_char_space_dots', 3.0);
        word_dots = p_vals.get('word_space_dots', 7.0);
        ramp_ms = p_vals.get('ramp_ms', 5)
        if char_wpm <= 0: self.status_var.set("Encoder Error: WPM must be > 0"); return None
        dot_s = 1.2 / char_wpm;
        dash_s = dot_s * dash_dot_ratio;
        elem_s_s = dot_s
        char_s_s = dot_s * inter_char_dots;
        word_s_s = dot_s * word_dots;
        ramp_samp = int(sample_rate * (ramp_ms / 1000.0))
        segments = [];
        first_in_seq = True

        def mk_ramp(n, up=True):
            r = np.linspace(0, 1, n, endpoint=False) if n > 0 else np.array([]); return r if up else 1 - r

        def gen_seg(d_s, tone):
            n_s = int(d_s * sample_rate);
            if n_s == 0: return np.array([])
            if tone:
                t = np.linspace(0, d_s, n_s, endpoint=False);
                w = volume * np.sin(2 * np.pi * frequency * t)
                rs = min(ramp_samp, n_s // 2)
                if rs > 0: w[:rs] *= mk_ramp(rs, True); w[-rs:] *= mk_ramp(rs, False)
                return w
            else:
                return np.zeros(n_s)

        for char_text in text:
            if char_text == ' ':
                if not first_in_seq: segments.append(gen_seg(word_s_s, False))
                first_in_seq = True;
                continue
            mc = MORSE_CODE_DICT_ENCODE.get(char_text, "")
            if not mc: print(f"Encoder: Skip unknown char: '{char_text}'"); continue
            if not first_in_seq: segments.append(gen_seg(char_s_s, False))
            first_in_seq = False
            for i, sym_morse in enumerate(mc):
                if sym_morse == '.':
                    segments.append(gen_seg(dot_s, True))
                elif sym_morse == '-':
                    segments.append(gen_seg(dash_s, True))
                if i < len(mc) - 1: segments.append(gen_seg(elem_s_s, False))
        if not segments: self.status_var.set("Encoder: No audio (empty/unknown chars)."); return None
        audio_f32 = np.concatenate(segments).astype(np.float32)
        return (audio_f32 * 32767).astype(np.int16), sample_rate

    def _encoder_play_thread_target(self, audio_data, sample_rate):
        self.encoder_is_playing = True
        if self.master.winfo_exists(): self.encoder_play_button.config(text="⏹ Stop"); self.status_var.set(
            "Encoder: Playing...")
        try:
            self.encoder_stream = self.pyaudio_instance.open(format=pyaudio.paInt16, channels=1, rate=sample_rate,
                                                             output=True)
            self.encoder_stream.write(audio_data.tobytes())
        except Exception as e:
            if self.master.winfo_exists(): self.status_var.set(f"Encoder Playback Error: {e}"); messagebox.showerror(
                "Encoder Playback Error", f"{e}")
        finally:
            if self.encoder_stream:
                try:
                    self.encoder_stream.stop_stream(); self.encoder_stream.close()
                except Exception:
                    pass
            self.encoder_stream = None;
            self.encoder_is_playing = False
            if self.master.winfo_exists():
                self.encoder_play_button.config(text="▶ Play")
                if "Error" not in self.status_var.get(): self.status_var.set("Encoder: Playback Finished.")
            self.encoder_playback_thread = None

    def encoder_toggle_play_stop(self):
        if self.encoder_is_playing:
            if self.encoder_stream:
                try:
                    self.encoder_stream.stop_stream()
                except Exception:
                    pass
            self.encoder_is_playing = False
            if self.master.winfo_exists(): self.encoder_play_button.config(text="▶ Play"); self.status_var.set(
                "Encoder: Playback Stopped.")
        else:
            audio_tuple = self.generate_morse_audio_data()
            if audio_tuple:
                audio_data, sample_rate = audio_tuple
                if self.encoder_playback_thread and self.encoder_playback_thread.is_alive():
                    if self.master.winfo_exists(): messagebox.showwarning("Busy", "Encoder already playing."); return
                self.encoder_playback_thread = threading.Thread(target=self._encoder_play_thread_target,
                                                                args=(audio_data, sample_rate))
                self.encoder_playback_thread.daemon = True;
                self.encoder_playback_thread.start()

    def encoder_save_wav(self):
        if self.encoder_is_playing:
            if self.master.winfo_exists(): messagebox.showwarning("Busy", "Stop playback before saving."); return
        audio_tuple = self.generate_morse_audio_data()
        if audio_tuple:
            audio_pcm, sample_rate = audio_tuple
            fp = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV files", "*.wav")])
            if fp:
                try:
                    with wave.open(fp, 'wb') as wf:
                        wf.setnchannels(1);
                        wf.setsampwidth(self.pyaudio_instance.get_sample_size(pyaudio.paInt16))
                        wf.setframerate(sample_rate);
                        wf.writeframes(audio_pcm.tobytes())
                    if self.master.winfo_exists(): self.status_var.set(
                        f"Encoder: Saved {os.path.basename(fp)}"); messagebox.showinfo("Save Successful",
                                                                                       f"Saved to:\n{fp}")
                except Exception as e:
                    if self.master.winfo_exists(): self.status_var.set(
                        f"Encoder Save Error: {e}"); messagebox.showerror("Save Error", f"{e}")
            elif self.master.winfo_exists():
                self.status_var.set("Encoder: Save cancelled.")

    def build_decoder_tab_ui(self):
        parent = self.decoder_tab
        file_frame = ttk.LabelFrame(parent, text="Input WAV File")
        file_frame.pack(padx=5, pady=10, fill="x")
        self.decoder_params['filepath'] = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.decoder_params['filepath'], width=50, state="readonly").grid(row=0,
                                                                                                             column=0,
                                                                                                             padx=(
                                                                                                             10, 5),
                                                                                                             pady=10,
                                                                                                             sticky="ew")
        ttk.Button(file_frame, text="Browse...", command=self.decoder_browse_file).grid(row=0, column=1, padx=5,
                                                                                        pady=10)
        file_frame.columnconfigure(0, weight=1)

        decoder_config_frame = ttk.LabelFrame(parent, text="Decoder Configuration")
        decoder_config_frame.pack(padx=5, pady=5, fill="x", expand=True)
        decoder_params_left = ttk.Frame(decoder_config_frame);
        decoder_params_left.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        decoder_params_right = ttk.Frame(decoder_config_frame);
        decoder_params_right.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        decoder_config_frame.columnconfigure(0, weight=1);
        decoder_config_frame.columnconfigure(1, weight=1)

        self.decoder_params['enable_noise_reduction'] = tk.BooleanVar(value=True)
        ttk.Checkbutton(decoder_params_left, text="Enable Noise Reduction",
                        variable=self.decoder_params['enable_noise_reduction']).pack(anchor="w", padx=5, pady=2)
        self.add_slider(decoder_params_left, self.decoder_params, 'nr_noise_clip_percentile', "NR Clip %:", 5, 50, 20,
                        width=18)
        self.add_combobox(decoder_params_left, self.decoder_params, 'nr_gain_type', "NR Gain Type:",
                          ['wiener_like_magnitude', 'wiener_like_power', 'simple_gate'], 'wiener_like_magnitude',
                          width=18)
        self.add_slider(decoder_params_left, self.decoder_params, 'nr_over_subtraction_factor', "NR Over-Sub:", 0.5,
                        2.0, 1.0, resolution=0.1, width=18)
        self.add_slider(decoder_params_left, self.decoder_params, 'dom_freq_min_hz', "DomFreqMin(Hz):", 50, 500, 150,
                        width=18)
        self.add_slider(decoder_params_left, self.decoder_params, 'dom_freq_max_hz', "DomFreqMax(Hz):", 1000, 3000,
                        2000, width=18)

        self.add_slider(decoder_params_right, self.decoder_params, 'bp_half_width_hz', "BP Half-Width(Hz):", 10, 200,
                        75, width=18)
        self.add_slider(decoder_params_right, self.decoder_params, 'bp_order', "BP Order:", 2, 8, 4, is_int=True,
                        width=18)
        self.add_slider(decoder_params_right, self.decoder_params, 'env_median_kernel_ms', "Env Kernel (ms):", 1, 50,
                        10, width=18)
        self.add_slider(decoder_params_right, self.decoder_params, 'threshold_std_factor', "Thresh Std Factor:", 0.1,
                        2.0, 0.6, resolution=0.1, width=18)
        self.add_slider(decoder_params_right, self.decoder_params, 'min_segment_duration_ms', "Min Seg (ms):", 5, 50,
                        15, width=18)
        self.add_slider(decoder_params_right, self.decoder_params, 'min_plausible_dot_ms', "Min Dot (ms):", 10, 100, 20,
                        width=18)

        decode_button_frame = ttk.Frame(parent);
        decode_button_frame.pack(pady=15)
        self.decode_button = ttk.Button(decode_button_frame, text="Decode Audio File",
                                        command=self.decoder_start_decoding, style='Accent.TButton')
        self.decode_button.pack(ipady=5, ipadx=10)

        results_frame = ttk.LabelFrame(parent, text="Decoder Results");
        results_frame.pack(padx=5, pady=10, fill="both", expand=True)
        ttk.Label(results_frame, text="Detected Morse:", font=('Segoe UI', 10, 'bold')).pack(anchor="w", padx=10,
                                                                                             pady=(10, 0))
        self.decoder_results['morse_text'] = scrolledtext.ScrolledText(results_frame, height=4, width=60, wrap=tk.WORD,
                                                                       font=('Consolas', 10))
        self.decoder_results['morse_text'].pack(padx=10, pady=5, fill="x", expand=True);
        self.decoder_results['morse_text'].configure(state='disabled')
        ttk.Label(results_frame, text="Decoded Text:", font=('Segoe UI', 10, 'bold')).pack(anchor="w", padx=10,
                                                                                           pady=(10, 0))
        self.decoder_results['decoded_text'] = scrolledtext.ScrolledText(results_frame, height=4, width=60,
                                                                         wrap=tk.WORD, font=('Segoe UI', 10))
        self.decoder_results['decoded_text'].pack(padx=10, pady=5, fill="x", expand=True);
        self.decoder_results['decoded_text'].configure(state='disabled')
        stats_frame = ttk.Frame(results_frame);
        stats_frame.pack(fill="x", padx=10, pady=10)
        self.decoder_results['wpm_var'] = tk.StringVar(value="WPM: --");
        ttk.Label(stats_frame, textvariable=self.decoder_results['wpm_var']).pack(side="left", padx=10)
        self.decoder_results['volume_var'] = tk.StringVar(value="Volume: --");
        ttk.Label(stats_frame, textvariable=self.decoder_results['volume_var']).pack(side="left", padx=10)
        self.decoder_results['farns_var'] = tk.StringVar(value="Farns Ratio: --");
        ttk.Label(stats_frame, textvariable=self.decoder_results['farns_var']).pack(side="left", padx=10)

    def decoder_browse_file(self):
        fp = filedialog.askopenfilename(title="Select WAV for Morse Decoding",
                                        filetypes=[("WAV files", "*.wav"), ("All files", "*.*")])
        if fp: self.decoder_params['filepath'].set(fp); self.status_var.set(f"Decoder: Loaded {os.path.basename(fp)}")

    def decoder_start_decoding(self):
        filepath = self.decoder_params['filepath'].get()
        if not filepath: messagebox.showerror("Input Error", "Please select a WAV file."); return
        if not os.path.exists(filepath): messagebox.showerror("File Error", f"File not found: {filepath}"); return
        if self.decoder_thread and self.decoder_thread.is_alive(): messagebox.showwarning("Busy",
                                                                                          "Decoder is running."); return

        self.status_var.set("Decoder: Processing...")
        self.decode_button.config(state="disabled")
        self.decoder_update_results("", "", "--", "--", "--")

        self.decoder_thread = threading.Thread(target=self._decoder_run_logic, args=(filepath,))
        self.decoder_thread.daemon = True;
        self.decoder_thread.start()

    def _decoder_run_logic(self, filepath):
        cfg = self.get_param_values(self.decoder_params)
        try:
            # Fixed values for params not yet in decoder GUI but needed by the logic function
            nr_n_fft_val = cfg.get('nr_n_fft', 1024)  # Example, if you add this slider later
            nr_hop_length_val = cfg.get('nr_hop_length', 256)  # Example
            nr_power_exponent_val = cfg.get('nr_power_exponent', 1.0)  # Example
            dom_typical_hz_val = cfg.get('dom_freq_typical_morse_hz', (400, 1200))
            thresh_min_signal_frac_val = cfg.get('threshold_min_signal_fraction', 0.005)
            thresh_fallback_perc_val = cfg.get('threshold_fallback_percentile', 75)
            thresh_abs_min_val = cfg.get('threshold_absolute_min', 0.02)
            expected_dash_dot_ratio_range_val = cfg.get('expected_dash_dot_ratio_range', (2.0, 4.0))

            morse, wpm, vol, farns = self._audio_to_morse_decoder_logic(
                filename=filepath,
                enable_noise_reduction=cfg.get('enable_noise_reduction', True),
                nr_n_fft=nr_n_fft_val,
                nr_hop_length=nr_hop_length_val,
                nr_noise_clip_percentile=cfg.get('nr_noise_clip_percentile', 20),
                nr_gain_type=cfg.get('nr_gain_type', 'wiener_like_magnitude'),
                nr_over_subtraction_factor=cfg.get('nr_over_subtraction_factor', 1.0),
                nr_power_exponent=nr_power_exponent_val,
                dom_freq_min_hz=cfg.get('dom_freq_min_hz', 150),
                dom_freq_max_hz=cfg.get('dom_freq_max_hz', 2000),
                dom_freq_typical_morse_hz=dom_typical_hz_val,
                bp_half_width_hz=cfg.get('bp_half_width_hz', 75),
                bp_order=cfg.get('bp_order', 4),
                env_median_kernel_ms=cfg.get('env_median_kernel_ms', 10),
                threshold_std_factor=cfg.get('threshold_std_factor', 0.6),
                threshold_min_signal_fraction=thresh_min_signal_frac_val,
                threshold_fallback_percentile=thresh_fallback_perc_val,
                threshold_absolute_min=thresh_abs_min_val,
                min_segment_duration_ms=cfg.get('min_segment_duration_ms', 15),
                min_plausible_dot_ms=cfg.get('min_plausible_dot_ms', 20),
                expected_dash_dot_ratio_range=expected_dash_dot_ratio_range_val
            )
            decoded_text_val = self._decode_morse_text(morse)
            if self.master.winfo_exists(): self.master.after(0, self.decoder_update_results, morse, decoded_text_val,
                                                             wpm, vol, farns)
        except Exception as e:
            error_msg = f"Decoder Error: {type(e).__name__}: {e}"
            print(f"Full Decoder Error Traceback:", file=sys.stderr)  # Print to stderr
            import traceback
            traceback.print_exc(file=sys.stderr)  # Print full traceback to stderr
            if self.master.winfo_exists():
                self.master.after(0, lambda: messagebox.showerror("Decoding Error", error_msg))
                self.master.after(0, self.decoder_update_results, "Error during decoding.", "", "--", "--", "--")
        finally:
            if self.master.winfo_exists():
                self.master.after(0, lambda: self.decode_button.config(state="normal"))
                current_status = self.status_var.get()
                final_status = "Decoder: Processing complete." if "Error" not in current_status else current_status
                self.master.after(0, lambda: self.status_var.set(final_status))

    def decoder_update_results(self, morse_code, decoded_text, wpm, volume, farns_ratio):
        for widget_key in ['morse_text', 'decoded_text']:
            if widget_key in self.decoder_results and self.decoder_results[widget_key].winfo_exists():
                self.decoder_results[widget_key].configure(state='normal')
                self.decoder_results[widget_key].delete(1.0, tk.END)

        if 'morse_text' in self.decoder_results and self.decoder_results['morse_text'].winfo_exists():
            self.decoder_results['morse_text'].insert(tk.END, morse_code if morse_code else "")
        if 'decoded_text' in self.decoder_results and self.decoder_results['decoded_text'].winfo_exists():
            self.decoder_results['decoded_text'].insert(tk.END, decoded_text if decoded_text else "")

        for widget_key in ['morse_text', 'decoded_text']:
            if widget_key in self.decoder_results and self.decoder_results[widget_key].winfo_exists():
                self.decoder_results[widget_key].configure(state='disabled')

        if 'wpm_var' in self.decoder_results and self.decoder_results['wpm_var']: self.decoder_results['wpm_var'].set(
            f"WPM: {wpm if isinstance(wpm, (int, float)) else '--'}")
        if 'volume_var' in self.decoder_results and self.decoder_results['volume_var']: self.decoder_results[
            'volume_var'].set(f"Volume: {volume if isinstance(volume, (int, float)) else '--'}")
        if 'farns_var' in self.decoder_results and self.decoder_results['farns_var']: self.decoder_results[
            'farns_var'].set(f"Farns Ratio: {farns_ratio if isinstance(farns_ratio, (int, float)) else '--'}")

        if self.master.winfo_exists():
            if not morse_code or "Error" in str(morse_code):
                if "Error" not in self.status_var.get(): self.status_var.set("Decoder: No Morse detected or error.")
            else:
                self.status_var.set("Decoder: Results updated.")

    def _decode_morse_text(self, morse_code_str):
        if not isinstance(morse_code_str, str): return ""
        words = morse_code_str.strip().split('   ')
        decoded_message = []
        for word in words:
            letters = word.strip().split(' ')
            decoded_word = ""
            for letter_code in letters:
                if letter_code: decoded_word += MORSE_CODE_DICT_DECODE.get(letter_code, '?')
            decoded_message.append(decoded_word)
        return ' '.join(decoded_message)

    def _audio_to_morse_decoder_logic(self, filename, enable_noise_reduction, nr_n_fft, nr_hop_length,
                                      nr_noise_clip_percentile, nr_gain_type, nr_over_subtraction_factor,
                                      nr_power_exponent, dom_freq_min_hz, dom_freq_max_hz,
                                      dom_freq_typical_morse_hz, bp_half_width_hz, bp_order,
                                      env_median_kernel_ms, threshold_std_factor, threshold_min_signal_fraction,
                                      threshold_fallback_percentile, threshold_absolute_min,
                                      min_segment_duration_ms, min_plausible_dot_ms,
                                      expected_dash_dot_ratio_range):
        try:
            rate, data_orig = scipy_wav.read(filename)
        except Exception as e:
            return f"Read Audio Error: {e}", 0, 0, 0

        if len(data_orig.shape) > 1: data_orig = data_orig.mean(axis=1)
        if data_orig.size < rate * 0.05: return "Audio too short", 0, 0, 0  # Min 50ms

        data = data_orig.astype(np.float32)
        if enable_noise_reduction:
            data = self._spectral_noise_reduction_logic(data, rate, nr_n_fft, nr_hop_length,
                                                        nr_noise_clip_percentile, nr_gain_type,
                                                        nr_over_subtraction_factor, nr_power_exponent)

        max_abs_val = np.max(np.abs(data))
        if max_abs_val < 1e-6: return "Audio silent post-NR", 0, 0, 0
        data_normalized = data / max_abs_val

        dominant_freq = self._detect_dominant_frequency_logic(data_normalized, rate,
                                                              dom_freq_min_hz, dom_freq_max_hz,
                                                              dom_freq_typical_morse_hz)
        if dominant_freq is None: return "No tone frequency", 0, 0, 0

        low_f = max(1.0, dominant_freq - bp_half_width_hz)
        high_f = min(rate / 2 - 1.0, dominant_freq + bp_half_width_hz)
        if low_f >= high_f: return f"BPF Error: low {low_f:.1f} >= high {high_f:.1f}", 0, 0, 0

        try:
            filtered_data = self._bandpass_filter_logic(data_normalized, rate, low_f, high_f, order=bp_order)
        except ValueError as e:
            return f"BPF Design Error: {e}", 0, 0, 0

        envelope = np.abs(filtered_data)
        kernel_s = max(3, int(rate * (env_median_kernel_ms / 1000.0)) * 2 + 1)
        smoothed_envelope = scipy_signal.medfilt(envelope, kernel_size=kernel_s)

        max_env_for_stats = np.max(smoothed_envelope) if smoothed_envelope.size > 0 else 0
        if max_env_for_stats < 1e-6:
            threshold = threshold_absolute_min
        else:
            env_for_stats = smoothed_envelope[smoothed_envelope > threshold_min_signal_fraction * max_env_for_stats]
            if len(env_for_stats) > 10:
                threshold = np.mean(env_for_stats) + threshold_std_factor * np.std(env_for_stats)
            else:
                threshold = np.percentile(smoothed_envelope, threshold_fallback_percentile) if len(
                    smoothed_envelope) > 0 else threshold_absolute_min
        threshold = max(threshold, threshold_absolute_min)
        tone_active = smoothed_envelope > threshold

        actual_min_segment_dur_sec = min_segment_duration_ms / 1000.0
        transitions = np.diff(tone_active.astype(int))
        event_indices = np.where(transitions != 0)[0] + 1
        segment_boundaries = np.unique(np.concatenate(([0], event_indices, [len(tone_active)])))
        if len(segment_boundaries) < 2: return "Segmentation Error (boundaries)", 0, 0, 0

        all_durs_s = np.diff(segment_boundaries) / rate
        all_types = tone_active[segment_boundaries[:-1]].astype(int)
        temp_durs = [d for d, t in zip(all_durs_s, all_types) if d >= actual_min_segment_dur_sec]
        temp_types = [t for d, t in zip(all_durs_s, all_types) if d >= actual_min_segment_dur_sec]
        if not temp_durs: return "No valid segments post-filter", 0, 0, 0

        merged_durs, merged_types = [temp_durs[0]], [temp_types[0]]
        for i in range(1, len(temp_durs)):
            if temp_types[i] == merged_types[-1]:
                merged_durs[-1] += temp_durs[i]
            else:
                merged_durs.append(temp_durs[i]); merged_types.append(temp_types[i])
        final_durs_sec, final_types = np.array(merged_durs), np.array(merged_types)
        if len(final_durs_sec) == 0: return "No segments post-merge", 0, 0, 0

        tone_durs = final_durs_sec[final_types == 1]
        silence_durs = final_durs_sec[final_types == 0]
        min_plausible_dot_sec = min_plausible_dot_ms / 1000.0
        tone_durs_for_cluster = tone_durs[tone_durs >= min_plausible_dot_sec]
        if len(
            tone_durs_for_cluster) < 2: return f"Not enough tones ({len(tone_durs_for_cluster)}) for clustering", 0, 0, 0

        _, (dot_dur, dash_dur) = cluster_durations_global(list(tone_durs_for_cluster), 2)  # Use global helper
        dot_threshold = (dot_dur + dash_dur) / 2
        if dot_dur <= 1e-3: return f"Dot too short ({dot_dur * 1000:.1f}ms)", 0, 0, 0

        ratio = dash_dur / dot_dur
        if not (expected_dash_dot_ratio_range[0] <= ratio <= expected_dash_dot_ratio_range[1]):
            print(f"[Decoder Logic WARNING] Dash/dot ratio ({ratio:.2f}) outside expected.")

        intra_gap, inter_char_gap, word_gap, inter_threshold, word_threshold = 0, 0, 0, 0, 0
        if len(silence_durs) == 0:
            intra_gap, inter_char_gap, word_gap = 0.5 * dot_dur, 3 * dot_dur, 7 * dot_dur
        else:
            num_distinct_s = len(np.unique(silence_durs))
            n_s_clusters = min(3, num_distinct_s, len(silence_durs))
            if n_s_clusters == 0: return "Silence cluster error (0 clusters)", 0, 0, 0
            _, s_centers = cluster_durations_global(list(silence_durs), n_s_clusters)  # Use global helper
            gaps = sorted(s_centers)
            if n_s_clusters == 1:
                intra_gap, inter_char_gap, word_gap = gaps[0], max(gaps[0] * 1.5, 3 * dot_dur), max(gaps[0] * 2.5,
                                                                                                    7 * dot_dur)
            elif n_s_clusters == 2:
                intra_gap, inter_char_gap, word_gap = gaps[0], gaps[1], max(gaps[1] * 1.5, 7 * dot_dur)
            else:
                intra_gap, inter_char_gap, word_gap = gaps[0], gaps[1], gaps[2]

        inter_threshold = (intra_gap + inter_char_gap) / 2 if abs(
            intra_gap - inter_char_gap) > 1e-3 else intra_gap + 0.5 * dot_dur
        word_threshold = (inter_char_gap + word_gap) / 2 if abs(
            inter_char_gap - word_gap) > 1e-3 else inter_char_gap + 2 * dot_dur
        if inter_threshold >= word_threshold and (len(silence_durs) > 0 and n_s_clusters > 1):
            word_threshold = inter_threshold + 2 * dot_dur
            if inter_threshold >= word_threshold: return "Silence threshold error", 0, 0, 0

        morse_str = ""
        last_was_tone = False
        first_tone_idx = next((i for i, t in enumerate(final_types) if t == 1), -1)
        if first_tone_idx == -1: return "No tones for reconstruction", 0, 0, 0

        for i in range(first_tone_idx, len(final_durs_sec)):
            dur, is_tone = final_durs_sec[i], (final_types[i] == 1)
            if is_tone:
                morse_str += ('.' if dur < dot_threshold else '-'); last_was_tone = True
            else:
                if last_was_tone:
                    if dur < inter_threshold:
                        pass
                    elif dur < word_threshold:
                        morse_str += ' '
                    else:
                        morse_str += '   '
                last_was_tone = False
        morse_str = morse_str.strip()

        wpm = round(1.2 / dot_dur) if dot_dur > 0 else 0
        farns = round(word_gap / dot_dur, 2) if dot_dur > 0 and word_gap > 0 else 0
        vol = round(np.max(np.abs(filtered_data)), 4) if 'filtered_data' in locals() and filtered_data.size > 0 else 0
        return morse_str, wpm, vol, farns

    def on_closing(self):
        if self.encoder_is_playing and self.encoder_stream:
            try:
                self.encoder_stream.stop_stream(); self.encoder_stream.close()
            except Exception:
                pass
        if self.pyaudio_instance:
            try:
                self.pyaudio_instance.terminate()
            except Exception:
                pass
        if self.encoder_playback_thread and self.encoder_playback_thread.is_alive():
            self.encoder_playback_thread.join(timeout=0.2)
        if self.decoder_thread and self.decoder_thread.is_alive():
            self.decoder_thread.join(timeout=0.2)
        self.master.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = MorseCodeApp(root)
    root.mainloop()