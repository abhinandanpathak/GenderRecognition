import sounddevice as sd

import scipy.io.wavfile
duration = 5  # seconds
fs=16000 #sampling_rate
sd.default.samplerate = fs #sets default sample_rate
sd.default.channels = 2 
myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
sd.wait() #waits for recording to end
scipy.io.wavfile.write('abhi.wav',fs,myrecording) #save as .wav file in current directory
