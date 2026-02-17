# 📡 Morse Code Toolkit Pro

A comprehensive Python application for encoding text to Morse code audio and decoding Morse code from WAV files back to readable text. Features advanced signal processing, noise reduction, and intelligent frequency detection.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 🎯 Features

### **Morse Code Encoder**

- Convert any text to Morse code audio signals
- Real-time audio playback with adjustable parameters
- Save generated audio as WAV files
- Customizable:
  - Character speed (WPM - Words Per Minute)
  - Tone frequency (200Hz - 10kHz)
  - Volume control
  - Dash/Dot ratio
  - Inter-character and word spacing
  - Audio ramp for smooth tone envelope

### **Morse Code Decoder**

- Extract Morse code from stored WAV files
- Advanced signal processing pipeline:
  - **Spectral Noise Reduction** with Wiener filtering
  - **Dominant Frequency Detection** using FFT
  - **Bandpass Filtering** around detected tone
  - **Envelope Detection** and adaptive thresholding
  - **K-means Clustering** for dot/dash discrimination
- Automatically estimates WPM, volume, and Farnsworth ratio
- Comprehensive configuration for fine-tuning

---

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or download the repository:**

   ```bash
   cd MorseCodeApp
   ```

2. **Install dependencies:**

   ```bash
   pip install -r Requirements.txt
   ```

   This installs:
   - `numpy` - Numerical computing and array operations
   - `PyAudio` - Audio input/output functionality
   - `scipy` - Advanced signal processing (filtering, STFT, wavfile I/O)
   - `scikit-learn` - Machine learning (K-means clustering for dot/dash detection)

3. **Run the application:**

   ```bash
   python app.py
   ```

---

## 📋 System Requirements

| Component | Requirement |
|-----------|-------------|
| **OS** | Windows, macOS, Linux |
| **Python** | 3.8+ |
| **RAM** | 512 MB minimum |
| **Audio** | System audio output device required |
| **Dependencies** | See Requirements.txt |

---

## 🚀 Usage

### **Encoding Text to Morse Code**

1. Open the **"Morse Encoder / Generator"** tab
2. Enter your message in the text field (default: "Jai Hind")
3. Adjust parameters:
   - **Character WPM**: 5-60 (words per minute at character level)
   - **Frequency**: 200-10000 Hz (tone pitch)
   - **Volume**: 0.0-1.0 (audio loudness)
   - **Ramp**: 0-50 ms (envelope rise/fall time)
   - **Dash/Dot Ratio**: 2.0-4.0 (typically 3.0)
   - **Char Space**: 1-10 dots (spacing between characters)
   - **Word Space**: 3-20 dots (spacing between words)
4. Click **▶ Play** to hear the audio
5. Click **💾 Save WAV** to save as file

#### Encoding Parameters Guide

**WPM (Words Per Minute)**: Standard Morse code timing. Higher values = faster transmission.

```
Dot duration = 1.2 / WPM seconds
Example: 20 WPM → Dot duration = 60 ms
```

**Dash/Dot Ratio**: How much longer dashes are than dots.

- Standard: 3.0 (dash = 3× dot duration)
- Range: 2.0 - 4.0

**Spacing**:

- Intra-element: Implicit (dot duration)
- Inter-character: `char_space_dots × dot_duration`
- Inter-word: `word_space_dots × dot_duration`

---

### **Decoding Morse Code from Audio**

1. Open the **"Morse Decoder (from WAV)"** tab
2. Click **Browse...** to select a WAV file
3. Configure decoder settings:

#### Noise Reduction

- **Enable Noise Reduction**: Toggle on/off
- **NR Clip %**: Percentile for noise floor estimation (5-50)
- **NR Gain Type**:
  - `wiener_like_magnitude`: Standard Wiener filter
  - `wiener_like_power`: Power-based variant
  - `simple_gate`: Binary thresholding
- **NR Over-Subtraction**: Factor for aggressive noise suppression (0.5-2.0)

#### Frequency Detection

- **DomFreqMin/Max**: Search range for tone frequency (50Hz - 3000Hz)

#### Filtering & Processing

- **BP Half-Width**: Bandwidth around detected frequency (10-200 Hz)
- **BP Order**: Filter steepness (2-8, higher = steeper)
- **Env Kernel**: Median filter window (1-50 ms)
- **Thresh Std Factor**: Sensitivity for tone detection threshold (0.1-2.0)

#### Segment Detection

- **Min Segment**: Minimum tone/silence duration (5-50 ms)
- **Min Dot**: Minimum plausible dot duration (10-100 ms)

1. Click **Decode Audio File**
2. Results show:
   - Detected Morse code pattern
   - Decoded text
   - Estimated WPM
   - Signal volume
   - Farnsworth ratio

---

## 🏗️ Architecture

### **System Architecture**

```
┌─────────────────────────────────────────────────────┐
│           Morse Code Toolkit Application            │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌────────────────────┐   ┌────────────────────┐   │
│  │  ENCODER TAB       │   │  DECODER TAB       │   │
│  ├────────────────────┤   ├────────────────────┤   │
│  │ Text Input         │   │ WAV File Input     │   │
│  │ Parameter Config   │   │ Signal Config      │   │
│  │ Preview & Save     │   │ Process & Results  │   │
│  └────────┬───────────┘   └────────┬───────────┘   │
│           │                        │                │
│           ▼                        ▼                │
│  ┌────────────────────┐   ┌────────────────────┐   │
│  │ Audio Generation   │   │ Signal Processing  │   │
│  │ & Playback Engine  │   │ & Decoding Engine  │   │
│  └────────┬───────────┘   └────────┬───────────┘   │
│           │                        │                │
│           ▼                        ▼                │
│  ┌────────────────────────────────────────────┐    │
│  │      Audio I/O (PyAudio + Wave)            │    │
│  │      DSP Pipeline (SciPy + NumPy)          │    │
│  │      ML (scikit-learn K-means)             │    │
│  └────────────────────────────────────────────┘    │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 📊 Signal Processing Pipeline

### **Encoding Pipeline**

```
Text Input
    ↓
Convert to Uppercase
    ↓
Character Lookup (Morse Dict)
    ↓
Generate Timing Segments
    ├─ Dot duration from WPM
    ├─ Dash duration from ratio
    └─ Spacing from configuration
    ↓
Waveform Generation
    ├─ Sinusoidal tone synthesis
    ├─ Amplitude envelope ramp
    └─ Volume scaling
    ↓
Concatenate Segments
    ↓
Convert to Audio Format (Int16)
    ↓
PlayBack OR Save WAV File
```

### **Decoding Pipeline**

```
WAV File Input
    ↓
Read Audio Data (SciPy)
    ↓
Mono Conversion (if stereo)
    ↓
┌─► SPECTRAL NOISE REDUCTION ──────────────┐
│   ├─ STFT decomposition                   │
│   ├─ Noise floor estimation               │
│   └─ Gain-based suppression               │
│                                           │
└─► Normalized Audio                       │
    ↓                                       │
    ├─► FFT Analysis ──────────────────────────┐
    │   ├─ Find dominant frequency             │
    │   └─ Verify in typical Morse range       │
    │                                          │
    └─► Bandpass Filtering ──────────────┐    │
        ├─ Design Butterworth filter     │    │
        └─ Apply forward-backward        │    │
                                         │    │
        ↓                                │    │
        ├─► Envelope Detection ◄─────────┘    │
        │   ├─ Magnitude of filtered signal   │
        │   ├─ Median smoothing               │
        │   └─ Adaptive thresholding          │
        │                                    │
        └─► Segmentation ──────────────────────┐
            ├─ Find tone/silence transitions   │
            ├─ Merge short segments           │
            └─ Extract durations              │
                                             │
            ↓                                │
            ├─► K-Means Clustering ◄─────────┘
            │   ├─ Cluster tone durations
            │   ├─ Identify dot & dash
            │   └─ Compute thresholds
            │
            └─► Silence Clustering
                ├─ Identify gaps
                ├─ Classify gaps (3-way)
                └─ Set timing thresholds
                    ↓
                ├─► Sequence Reconstruction
                │   ├─ Convert tones to dots/dashes
                │   ├─ Insert space markers
                │   └─ Generate Morse string
                │
                ├─► Morse to Text Conversion
                │   └─ Look up Morse dictionary
                │
                └─► Metrics Calculation
                    ├─ WPM estimation
                    ├─ Volume (peak)
                    └─ Farnsworth ratio
                        ↓
                    Return: (Morse, WPM, Volume, Farns)
```

---

## 🏛️ Class Structure

### **UML Class Diagram**

```
┌──────────────────────────────────────────────────────┐
│                  MorseCodeApp                        │
├──────────────────────────────────────────────────────┤
│ ATTRIBUTES:                                          │
│ ├─ master: tk.Tk                                    │
│ ├─ pyaudio_instance: PyAudio                        │
│ ├─ encoder_*: playback control variables            │
│ ├─ decoder_*: decoding state variables              │
│ ├─ encoder_params: Dict[str, tk.Variable]           │
│ ├─ decoder_params: Dict[str, tk.Variable]           │
│ ├─ decoder_results: Dict[str, tk.Widget]            │
│ ├─ status_var: tk.StringVar                         │
│ └─ notebook: ttk.Notebook                           │
├──────────────────────────────────────────────────────┤
│ METHODS:                                             │
│                                                      │
│ ▶ GUI Building:                                     │
│ ├─ build_encoder_tab_ui()                           │
│ ├─ build_decoder_tab_ui()                           │
│ ├─ add_slider(...)       [UI Helper]                │
│ └─ add_combobox(...)     [UI Helper]                │
│                                                      │
│ ▶ Encoder Methods:                                  │
│ ├─ generate_morse_audio_data() → (audio, sr)        │
│ ├─ encoder_toggle_play_stop()                       │
│ ├─ encoder_save_wav()                               │
│ └─ _encoder_play_thread_target(audio, sr)           │
│                                                      │
│ ▶ Decoder Methods:                                  │
│ ├─ decoder_browse_file()                            │
│ ├─ decoder_start_decoding()                         │
│ └─ _decode_processing_thread_target()               │
│                                                      │
│ ▶ DSP Methods (Decoder):                            │
│ ├─ _spectral_noise_reduction_logic(...)             │
│ ├─ _bandpass_filter_logic(...)                      │
│ ├─ _detect_dominant_frequency_logic(...)            │
│ ├─ _audio_to_morse_decoder_logic(...)               │
│ └─ _decode_morse_text(morse_str) → str              │
│                                                      │
│ ▶ Utility Methods:                                  │
│ ├─ get_param_values(dict) → Dict                    │
│ ├─ _update_slider_label(...)                        │
│ ├─ _update_decoder_results(...)                     │
│ └─ on_closing()                                     │
└──────────────────────────────────────────────────────┘
```

---

## 📈 Data Flow Diagrams

### **Encoding Data Flow**

```
User Input (Text)
    │
    ├─ Parameter Dict (WPM, Freq, Vol, etc.)
    ├─ Morse Code Dictionary
    │
    ▼ generate_morse_audio_data()
┌─────────────────────────────┐
│  Timing Calculations        │
│  ├─ dot_duration (s)        │
│  ├─ dash_duration (s)       │
│  ├─ element_spacing (s)     │
│  ├─ char_spacing (s)        │
│  └─ word_spacing (s)        │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  Tone Generation            │
│  ├─ Sinusoid synthesis      │
│  ├─ Envelope ramping        │
│  └─ Volume scaling          │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  Sample Array (np.float32)  │
│  (Floating point samples)   │
└─────────────────────────────┘
    │
    ├─ Path 1: Playback
    │   ├─ Convert to Int16
    │   ├─ Open PyAudio stream
    │   └─ Stream to speakers
    │
    └─ Path 2: Save to File
        ├─ Convert to Int16
        ├─ Open Wave file
        └─ Write frames + metadata
```

### **Decoding Data Flow**

```
User: Select WAV File
    │
    ▼
Load Audio File (scipy_wav.read)
    ├─ Sample rate (rate)
    ├─ Audio samples (data_orig)
    │
    ▼
Mono Conversion (if stereo)
    │
    ▼ [Optional] NOISE REDUCTION
    │ _spectral_noise_reduction_logic()
    │ ├─ STFT decomposition
    │ ├─ Noise profile estimation
    │ ├─ Gain computation
    │ └─ iSTFT reconstruction
    │
    ▼
Normalize Audio
    │
    ▼ FREQUENCY DETECTION
    │ _detect_dominant_frequency_logic()
    │ ├─ FFT analysis
    │ ├─ Search in [min, max] Hz range
    │ ├─ Prefer typical Morse range
    │ └─ Return: dominant_freq
    │
    ▼ BANDPASS FILTERING
    │ _bandpass_filter_logic()
    │ ├─ Design Butterworth filter
    │ │  (centered on dominant freq)
    │ └─ Apply filtfilt (forward-backward)
    │
    ▼
Envelope Detection
    ├─ Magnitude of analytic signal
    ├─ Median smoothing
    │
    ▼
Adaptive Thresholding
    ├─ Percentile + std-dev based
    ├─ Determine tone/silence boundary
    │
    ▼
Segmentation
    ├─ Find state transitions
    ├─ Merge short segments
    ├─ Filter by minimum duration
    │
    ▼ K-MEANS CLUSTERING
    │ cluster_durations_global()
    │ ├─ Cluster tone durations → (dot, dash)
    │ ├─ Dot = short cluster center
    │ └─ Dash = long cluster center
    │
    ▼ SILENCE CLUSTERING
    │ cluster_durations_global()
    │ ├─ Cluster silence durations
    │ ├─ Classify: intra-elem, inter-char, word
    │
    ▼
Morse Code Assembly
    ├─ Tones < dot_threshold → '.'
    ├─ Tones ≥ dot_threshold → '-'
    ├─ Silences: insert ' ' or '   '
    │
    ▼
Text Decoding
    └─ Lookup Morse Dictionary
        ├─ Word-by-word parsing
        ├─ Character-by-character conversion
        │
        ▼
RESULTS:
├─ Morse code string
├─ Decoded text
├─ Estimated WPM
├─ Signal volume (peak)
└─ Farnsworth ratio (word spacing factor)
```

---

## 🔑 Key Algorithms

### **1. Morse Code Encoding**

```
Algorithm: TEXT_TO_MORSE_AUDIO

Input:
  - text: String to encode
  - char_wpm: Character transmission speed
  - frequency: Tone frequency (Hz)
  - volume: Amplitude (0-1)
  - sample_rate: Audio sample rate (Hz)

Process:
  1. Calculate timing:
     dot_duration = 1.2 / char_wpm
     dash_duration = dot_duration × dash_ratio
     
  2. For each character in text:
     a. Look up Morse code pattern
     b. For each symbol (. or -):
        - Generate sinusoidal tone
        - Apply amplitude envelope ramp
        - Scale by volume
     c. Add inter-element silence
     d. Add inter-character silence (after char)
     e. Add inter-word silence (after space)
  
  3. Concatenate all audio segments
  4. Convert float32 → int16 (PCM)

Output: (audio_samples, sample_rate)
```

### **2. Morse Code Decoding (Multi-stage)**

#### **Stage 1: Audio Analysis**

```
Algorithm: DETECT_DOMINANT_FREQUENCY

Input:
  - data: Audio signal (float)
  - rate: Sample rate (Hz)
  - min_freq, max_freq: Search bounds
  - typical_morse_range: Preferred range

Process:
  1. Remove DC component (subtract mean)
  2. Compute FFT:
     - fft_values = FFT(data)
     - fft_freqs = frequency bins
     - fft_magnitude = |fft_values|
  
  3. Extract valid frequency range [min_freq, max_freq]
  
  4. Find peak magnitude:
     - Best attempt in preferred Morse range [400-1000 Hz]
     - Fallback to overall best if preferred fails
  
  5. Return peak frequency

Output: dominant_freq (Hz)
```

#### **Stage 2: Segmentation**

```
Algorithm: SEGMENT_AUDIO

Input:
  - filtered_audio: Bandpass-filtered signal
  - rate: Sample rate
  - threshold: Tone detection threshold

Process:
  1. Compute envelope:
     - envelope = |filtered_audio|
     - smooth_envelope = medfilt(envelope, kernel)
  
  2. Detect tone activity:
     - tone_active = (envelope > threshold)
  
  3. Find transitions:
     - transitions = diff(tone_active)
     - event_indices = where(transitions ≠ 0)
  
  4. Extract segment boundaries:
     - boundaries = transition points
     - durations = (boundary[i+1] - boundary[i]) / rate
  
  5. Classify segments as tone or silence
  
  6. Merge adjacent same-type segments
  
  7. Filter by minimum duration threshold

Output: (segment_durations, segment_types)
```

#### **Stage 3: K-means Clustering**

```
Algorithm: CLUSTER_TONE_DURATIONS

Input:
  - tone_durations: Array of tone segment durations
  - n_clusters: 2 (for dot & dash)

Process:
  1. Prepare data:
     - data = reshape(durations) → column vector
  
  2. Run K-means:
     - Fit K-means with k=2, random_state=0
     - centers = cluster centers (sorted)
  
  3. Interpret results:
     - center[0] → dot duration (shorter)
     - center[1] → dash duration (longer)
     - dot_threshold = (center[0] + center[1]) / 2

Output: (cluster_labels, (dot_dur, dash_dur))
```

#### **Stage 4: Sequence Reconstruction**

```
Algorithm: RECONSTRUCT_MORSE_SEQUENCE

Input:
  - segment_durations: Time durations
  - segment_types: Tone (1) or silence (0)
  - dot_dur, dash_dur: From clustering
  - gap_thresholds: intra, inter_char, word

Process:
  For each segment:
    If is_tone:
      - dur < dot_threshold → append '.'
      - dur ≥ dot_threshold → append '-'
    
    Else if is_silence:
      - dur < inter_threshold → (intra-element, no mark)
      - dur < word_threshold → append ' ' (inter-char)
      - dur ≥ word_threshold → append '   ' (inter-word)

Output: morse_code_string
```

#### **Stage 5: Morse to Text**

```
Algorithm: DECODE_MORSE_TO_TEXT

Input:
  - morse_string: Encoded morse (dots, dashes, spaces)

Process:
  1. Split by '   ' (word separator)
  2. For each word:
     a. Split by ' ' (char separator)
     b. Look up each code in Morse dictionary
     c. Append character to word
  3. Join words with spaces

Output: decoded_text
```

---

## ⚙️ Parameter Reference

### **Encoding Parameters**

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| char_wpm | 5-60 | 20 | Words per minute (character level) |
| frequency_hz | 200-10000 | 700 | Tone frequency in Hz |
| volume | 0.0-1.0 | 0.67 | Audio amplitude (0=silent, 1=max) |
| ramp_ms | 0-50 | 5 | Envelope rise/fall time in ms |
| dash_dot_ratio | 2.0-4.0 | 3.0 | Duration ratio (dash/dot) |
| inter_char_space_dots | 1-10 | 3 | Character spacing in dot units |
| word_space_dots | 3-20 | 7 | Word spacing in dot units |
| sample_rate_hz | [8k, 16k, 22.05k, 44.1k, 48k] | 44100 | Audio sample rate |

### **Decoding Parameters**

| Parameter | Range | Default | Function |
|-----------|-------|---------|----------|
| **Noise Reduction** |
| enable_noise_reduction | on/off | on | Toggle spectral noise reduction |
| nr_noise_clip_percentile | 5-50 | 20 | Percentile for noise floor |
| nr_gain_type | 3 options | wiener_like_magnitude | Filter type |
| nr_over_subtraction_factor | 0.5-2.0 | 1.0 | Noise suppression aggressiveness |
| **Frequency Detection** |
| dom_freq_min_hz | 50-500 | 150 | Minimum search frequency |
| dom_freq_max_hz | 1000-3000 | 2000 | Maximum search frequency |
| **Bandpass Filter** |
| bp_half_width_hz | 10-200 | 75 | Filter bandwidth (±Hz from center) |
| bp_order | 2-8 | 4 | Filter steepness (higher=steeper) |
| **Envelope Detection** |
| env_median_kernel_ms | 1-50 | 10 | Smoothing kernel size |
| threshold_std_factor | 0.1-2.0 | 0.6 | Threshold sensitivity |
| **Segmentation** |
| min_segment_duration_ms | 5-50 | 15 | Minimum segment length |
| min_plausible_dot_ms | 10-100 | 20 | Minimum dot duration |

---

## 🐛 Troubleshooting

### **Encoder Issues**

| Problem | Solution |
|---------|----------|
| No sound output | Check volume setting (>0.0) and system audio device |
| Audio is distorted | Reduce volume slider or lower WPM value |
| File not saving | Ensure write permissions in target directory |
| Morse code appears wrong | Check character support (alphanumeric + punctuation) |

### **Decoder Issues**

| Problem | Solution |
|---------|----------|
| "No tone frequency" error | Audio file may contain noise only. Check file quality. |
| Poor decoding accuracy | Try adjusting noise reduction parameters or using cleaner audio |
| WPM shows as 0 | Audio may be too silent. Check input file and try enabling NR. |
| Segments not detected | Reduce `min_segment_duration_ms` or increase `threshold_std_factor` |
| Decoder hangs | Large files may take time. Wait or reduce file size. |

### **Installation Issues**

| Problem | Solution |
|---------|----------|
| PyAudio fails to install | Use pre-built wheels or install PortAudio development files |
| scikit-learn errors | Ensure NumPy is installed first |
| Tkinter not found | Install `python3-tk` (Ubuntu/Debian) or use official Python distribution |

---

## 📦 Dependencies Breakdown

```
numpy~=2.2.6
  └─ Numerical arrays, linear algebra, signal operations
     ├─ Used in: Audio buffer management, FFT computation
     └─ Critical: Yes

PyAudio~=0.2.14
  └─ Audio input/output interface (cross-platform)
     ├─ Used in: Real-time playback, stream management
     └─ Critical: Yes (for playback)

scipy~=1.15.3
  └─ Scientific computing (signal processing, I/O)
     ├─ scipy.signal: IIR filters, STFT, envelope detection
     ├─ scipy.io.wavfile: WAV file reading/writing
     └─ Critical: Yes (for decoding)

scikit-learn~=1.5.x (implicit via scipy scipy)
  └─ Machine learning library
     ├─ sklearn.cluster.KMeans: Dot/dash clustering
     └─ Critical: Yes (for intelligent discrimination)

tkinter (built-in)
  └─ GUI framework (included with Python)
     ├─ Used in: All user interface
     └─ Critical: Yes
```

---

## 🎓 How It Works

### **Morse Code Alphabet**

Common Morse patterns:

```
A: .-     B: -...   C: -.-.   D: -..    E: .
F: ..-.   G: --.    H: ....   I: ..     J: .---
K: -.-    L: .-..   M: --     N: -.     O: ---
...and more with numbers and punctuation
```

### **Timing in Morse Code**

Standard PARIS convention (used for WPM calculation):

```
Dot duration:        1 unit
Dash duration:       3 units
Intra-element gap:   1 unit
Inter-character gap: 3 units
Inter-word gap:      7 units (or 5 in Farnsworth mode)

WPM = 1200 / dot_duration(ms) [PARIS calibration]
```

In this app:

```
dot_duration = 1.2 / WPM  [seconds]
dash_duration = dot_duration × dash_ratio (typically 3.0)
```

---

## 📝 File Structure

```
MorseCodeApp/
├── app.py                    # Main application
├── Requirements.txt          # Python dependencies
├── README.md                # This file
└── .gitignore              # Git ignore rules
```

---

## 🔬 Advanced: Understanding Signal Processing

### **Spectral Noise Reduction (Wiener Filter)**

The decoder employs an adaptive spectral subtraction method:

1. **STFT**: Convert time-domain audio to time-frequency representation
2. **Noise Estimation**: Estimate noise floor from quiet portions
3. **Gain Computation**: Calculate suppression gain per frequency bin
4. **iSTFT**: Convert back to time domain

```
Gain(freq) = max(0, 1 - α × N(freq) / S(freq))
where: S(freq) = signal spectrum
       N(freq) = estimated noise spectrum
       α = over-subtraction factor
```

### **Bandpass Filtering (Butterworth)**

Isolates the detected tone frequency:

```
Design: Butterworth IIR filter
Order: 2-8 (user configurable)
Range: dominant_freq ± bp_half_width
Method: Forward-backward (zero-phase distortion)
```

---

## 🚦 Status & Roadmap

### **Current Version Features** ✅

- [x] Text-to-Morse encoding with audio playback
- [x] WAV file save capability
- [x] Morse-to-text decoding from audio
- [x] Advanced noise reduction
- [x] Automatic frequency detection
- [x] Intelligent dot/dash discrimination
- [x] Real-time parameter adjustment

### **Potential Future Enhancements**

- [ ] Morse code transmission via radio frequencies
- [ ] Support for CW (continuous wave) keyer input
- [ ] Audio visualization (spectrograms, waveforms)
- [ ] Batch file processing
- [ ] International Morse code variants
- [ ] Export to MIDI or other formats

---

## 📬 Contributing

Contributions are welcome! Please:

1. Test your changes thoroughly
2. Follow PEP 8 style guidelines for Python code
3. Document any new features or parameters
4. Update this README if needed

---

## 📄 License

This project is provided as-is for educational and amateur radio purposes. Morse code transmission regulations vary by region—ensure compliance with local communications laws.

---

## 👤 Author

**fl4nk3r**  
Educational Morse Code Toolkit for Learning and Practice

---

## 📧 Support

For issues, questions, or feedback:

- Check the **Troubleshooting** section above
- Review decoder/encoder parameter settings
- Ensure all dependencies are correctly installed
- Test with sample WAV files first

---

**Last Updated:** February 17, 2026  
**Version:** 1.0.0  
**Stability:** Production Ready

---

## 🙏 Acknowledgments

- **SciPy**: Advanced signal processing
- **NumPy**: Numerical computing backbone
- **scikit-learn**: Machine learning clustering
- **PyAudio**: Cross-platform audio I/O
- **Tkinter**: User interface framework
