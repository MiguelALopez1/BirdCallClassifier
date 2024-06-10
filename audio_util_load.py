import torchaudio
from audio_util import AudioUtil

# Example usage of the AudioUtil class
audio_file = 'path/to/audio/file.wav'
aud = AudioUtil.open(audio_file)
reaud = AudioUtil.resample(aud, 44100)
rechan = AudioUtil.rechannel(reaud, 2)
dur_aud = AudioUtil.pad_trunc(rechan, 4000)
shift_aud = AudioUtil.time_shift(dur_aud, 0.4)
sgram = AudioUtil.spectro_gram(shift_aud)
aug_sgram = AudioUtil.spectro_augment(sgram)

print(aug_sgram.shape)
