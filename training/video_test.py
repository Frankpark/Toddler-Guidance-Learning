from moviepy.editor import *
from moviepy.audio.AudioClip import AudioArrayClip
import numpy as np
import math

# 15fps
frames = np.concatenate([np.ones([15, 84, 84, 3]), np.zeros([15, 84, 84, 3]), np.ones([15, 84, 84, 3])], axis = 0)
audioL = np.array([math.sin(2*math.pi*440*i/44100) for i in range(44100 * 3)])
audioR = np.array([math.sin(2*math.pi*660*i/44100) for i in range(44100 * 3)])
audios = np.stack([audioL, audioR], axis = 0)
print(frames.shape, audios.shape)

# Images
frames = [255 * frames[i] for i in range(frames.shape[0])]
#frames = 255 * frames
image_clip = ImageSequenceClip(frames, fps=15)

# Audios
#audios = (audios * 32768).astype(np.int16)
audios = np.transpose(audios)
print(audios.shape, audios.min(), audios.max())
audioclip = AudioArrayClip(audios, fps=44100)
#audioclip.write_audiofile('audio.mp3')

# Make video
video_clip = image_clip.set_audio(audioclip)
video_clip.write_videofile("result.mp4", fps=15, temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac")
