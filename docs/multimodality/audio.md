# Audio, Speech to Text, Text to Speech, Specch to Speech

- [Concepts]()
- [Speech-to-Text (STT) Models]()
  - [Whisper]()
  - [Fine-tune Whisper]()
- [Text-to-Speech (TTS) Models]()
  - ?
- [Speech-to-Speech Models]()
  - ?
- [Realworld Use Cases]()

## Concepts
### What Does Frequency Mean in Audio?

A sound wave propagates through the air by expanding and contracting air particles. The expansion part of this process is known as **rarefaction**, while the contraction is called **compression**. A sound wave rapidly oscillates between these two states, back and forth, over and over. A single one of these oscillations is called a **wave cycle**.

<img src="https://www.headphonesty.com/wp-content/uploads/2024/02/A_diagram_showing_compression__rarefaction__and_a_wave_cycle_in_a_sound_wave.jpg" height="30%" width="30%" />

Frequency refers to the number of wave cycles undergone by a given sound wave over the course of one second. The unit **hertz** (Hz) is used when measuring frequencies, which simply denotes the cycles per second.

Generate wave plot using python?

```python
import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inlinesignal, sr = librosa.load(librosa.util.example(‘brahms’))

# Display wave plot
plt.figure(figsize=(20, 5))
librosa.display.waveplot(signal, sr=sr)
plt.title(‘Waveplot’, fontdict=dict(size=18))
plt.xlabel(‘Time’, fontdict=dict(size=15))
plt.ylabel(‘Amplitude’, fontdict=dict(size=15))
plt.show()
```

<img src="https://miro.medium.com/v2/resize:fit:1100/format:webp/1*nXcScGcjm63ko30suhqvQA.png" height="70%" width="70%" />

### How to interpret a spectrogram / mel-spectogram?

A spectrogram is a graphical representation where:
— The horizontal axis corresponds to time.
— The vertical axis represents frequencies.
— The color intensities indicate the amplitude at each frequency and at each instant.

<img src="https://miro.medium.com/v2/resize:fit:1100/format:webp/1*0BQgSFuoo6HMrNtBnXjYLw.png" height="60%" width="60%" />

Above x-axis is time, y-axis (frequency) to log scale, and the “color” axis (amplitude) to Decibels, which is kinda the log scale of amplitudes.

### How to create spectrograms from raw audio?

Spectograms are created using Fast Fourier Transform (FFT) function that converts audio signal in the time domain into frequency domain. The magnitude of each frequency present in the audio is plotted onto a graph.

You can apply CNN algorithms on Sepctograms. 

The difference between a spectrogram and a Mel-spectrogram is that a Mel-spectrogram converts the frequencies to the mel-scale. The mel-scale is “a perceptual scale of pitches judged by listeners to be equal in distance from one another”

## Speech-to-Text Models

### OpenAI Whisper

<img src=https://cdn-images-1.medium.com/v2/resize:fit:800/1*aSdK_bRq3bhrXpP_TdgIHg.png" />

:star::star::star: [Decoding Whisper: An In-Depth Look at its Architecture and Transcription Process](https://amgadhasan.substack.com/p/whisper-how-to-create-robust-asr-46b)

- [Audio Deep Learning Made Simple: Sound Classification, Step-by-Step - An end-to-end example and architecture for audio deep learning’s foundational application scenario, in plain English](https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5)
- 
### Fine-tune Whisper
- [Fine-tune OpenAI’s Whisper Automatic Speech Recognition (ASR) Model](https://medium.com/graphcore/fine-tune-openais-whisper-automatic-speech-recognition-asr-model-394b5a4838fb)

## Real-world Use Cases
- [AI Powered Call Center Intelligence Accelerator](https://github.com/amulchapla/AI-Powered-Call-Center-Intelligence)
