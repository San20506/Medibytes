from transformers import pipeline

# Load the automatic speech recognition pipeline with MedASR
pipe = pipeline("automatic-speech-recognition", model="google/medasr")

# Transcribe your audio file
result = pipe("path_to_medical_audio.wav", chunk_length_s=20, stride_length_s=2)
print(result)
