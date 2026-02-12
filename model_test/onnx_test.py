import librosa
import numpy as np
import onnxruntime as ort
import torch
from madmom.audio.filters import LogarithmicFilterbank
from madmom.audio.spectrogram import Spectrogram, LogarithmicFilteredSpectrogram
from madmom.audio.signal import Signal
import re
import os

def sort_files(l):
    """ Sorts the given iterable numerically"""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def get_log_filterbank_frequencies(sample_rate, frame_size, num_bands=24, fmin=65, fmax=2100):
    frequencies = np.fft.rfftfreq(frame_size, d=1/sample_rate)
    filt = LogarithmicFilterbank(frequencies, num_bands=num_bands, fmin=fmin, fmax=fmax)
    
    # 提取每個頻帶的中心頻率
    center_frequencies = filt.center_frequencies
    return center_frequencies

def compute_log_spectrogram(audio_path, sample_rate=48000, frame_size=8192, hop_size=8192):
    signal = Signal(audio_path, sample_rate=sample_rate, num_channels=1)  # 確保單通道
    spec = Spectrogram(signal, frame_size=frame_size, hop_size=hop_size)
    frequencies = np.fft.rfftfreq(frame_size, d=1/sample_rate)
    
    # 確保頻率數組與頻譜形狀匹配
    frequencies = frequencies[:spec.shape[-1]]
    
    # 初始化對數濾波器組
    filt = LogarithmicFilterbank(frequencies, num_bands=24, fmin=65, fmax=2100)
    
    # 應用濾波器並計算對數頻譜
    log_filt_spec = LogarithmicFilteredSpectrogram(spec, filterbank=filt)
    log_spec = np.log1p(log_filt_spec)
    
    # 調換維度，將時間維度放在最後
    log_spec = log_spec.T  # (時間, 頻率) -> (頻率, 時間)
    return log_spec

def cyclic_pad_segment(segment, target_length):
    """Pads the segment cyclically until it reaches the target length."""
    current_length = segment.shape[3]  # Time dimension (last dimension)
    if current_length >= target_length:
        return segment[:, :, :, :target_length]  # Return sliced segment
    else:
        repeat_count = (target_length // current_length) + 1
        # Repeat only in the time dimension (last dimension)
        padded_segment = segment.repeat(1, 1, 1, repeat_count)  # Repeat in the time axis
        return padded_segment[:, :, :, :target_length]  # Trim the result to the target length


def load_and_preprocess_audio(audio_path, sample_rate=48000, frame_size=8192, hop_size=8192, target_length=175):
    """Computes the log spectrogram for the given audio file and pads it cyclically if necessary."""
    log_spec = compute_log_spectrogram(audio_path, sample_rate, frame_size, hop_size)
    
    # Normalize the spectrogram
    min_val = log_spec.min()
    max_val = log_spec.max()
    log_spec_normalized = (log_spec - min_val) / (max_val - min_val)
    
    # Reshape to match the input format expected by the model (batch_size, channels, height, width)
    log_spec_normalized = torch.from_numpy(log_spec_normalized).unsqueeze(0).unsqueeze(0)  # [1, 1, frequency, time]
    
    # Apply cyclic padding if the time dimension is less than the target length
    if log_spec_normalized.shape[3] < target_length:
        log_spec_normalized = cyclic_pad_segment(log_spec_normalized, target_length)
    
    return log_spec_normalized.numpy()  # Convert to NumPy for ONNX input


def classify_audio_onnx(onnx_model_path, audio_path, label_mapping, sample_rate=48000, frame_size=8192, hop_size=8192, target_length=175):
    # Reverse the label mapping for decoding the output
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}
    
    # Load ONNX model and create an inference session
    ort_session = ort.InferenceSession(onnx_model_path)
    
    # Prepare the spectrogram input from the audio file with cyclic padding
    input_spectrogram = load_and_preprocess_audio(audio_path, sample_rate, frame_size, hop_size, target_length)
    
    # Get the model's input name and prepare the input dictionary
    input_name = ort_session.get_inputs()[0].name
    input_data = {input_name: input_spectrogram}
    
    # Run inference
    outputs = ort_session.run(None, input_data)
    
    # Extract the predicted label (output is a list of arrays)
    predicted_label = np.argmax(outputs[0], axis=1)[0]  # Get the index of the highest score
    
    # Map the predicted label to the musical key
    predicted_key = reverse_label_mapping[predicted_label]
    return predicted_key


# Define label mapping
label_mapping = {
    'C major': 0, 'A minor': 0,
    'C# major': 1, 'D♭ major': 1, 'A# minor': 1, 'B♭ minor': 1,
    'D major': 2, 'B minor': 2,
    'E♭ major': 3, 'D# major': 3, 'C minor': 3,
    'E major': 4, 'C# minor': 4, 'D♭ minor': 4,
    'F major': 5, 'D minor': 5,
    'F# major': 6, 'G♭ major': 6, 'D# minor': 6, 'E♭ minor': 6,
    'G major': 7, 'E minor': 7,
    'A♭ major': 8, 'G# major': 8, 'F minor': 8,
    'A major': 9, 'F# minor': 9, 'G♭ minor': 9,
    'B♭ major': 10, 'A# major': 10, 'G minor': 10,
    'B major': 11, 'G# minor': 11, 'A♭ minor': 11
}

# Run classification on a real audio file using the ONNX model
onnx_model_path = "tonality_model.onnx"
# audio_file = "test_audio/instrument_estimated_43.wav"
# predicted_key = classify_audio_onnx(onnx_model_path, audio_file, label_mapping)
# print(f"The predicted key is: {predicted_key}")


''' Loop through test_audio folder '''
test_folder = 'test_audio/'
# Get a list of file names within the directory
files = os.listdir(test_folder)
sorted_files = sort_files(files)

# Loop through each file in the folder
for i, filename in enumerate(sorted_files):

    audio_path = os.path.join(test_folder, filename)
    print(f'\nProcessing file: {audio_path}')
    predicted_key = classify_audio_onnx(onnx_model_path, audio_path, label_mapping)
    print(f"The predicted key is: {predicted_key}")



