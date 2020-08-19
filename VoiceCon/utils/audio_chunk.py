from pydub import AudioSegment
import math
from os import path

class SplitWavAudioMubin():
    def __init__(self, folder, filename):
        self.folder = folder
        self.filename = filename
        self.filepath = path.join(folder, filename)

        self.audio = AudioSegment.from_wav(self.filepath)

    def get_duration(self):
        return self.audio.duration_seconds

    def single_split(self, from_sec, to_sec, split_filename):
        # t1 = from_min * 60 * 1000
        # t2 = to_min * 60 * 1000
        t1 = from_sec * 1000
        t2 = to_sec * 1000
        split_audio = self.audio[t1:t2]
        split_audio.export(path.join(self.folder, split_filename), format="wav")

    def multiple_split(self, seq_lengths=None):
        total_secs = self.get_duration()
        if seq_lengths is None:
            seq_lengths = [3, 5, 10, 15, 17, 20, 30, 40, 55, 70]
        elif type(seq_lengths) is str:
            seq_lengths = [seq_lengths]
        splits = []
        # for i in range(0, total_mins, seq_lengths):
        for i in seq_lengths:
            if i <= total_secs:
                split_fn = str(i) + '_' + self.filename
                from_min = 0
                self.single_split(from_min, from_min+i, split_fn)
                print(str(i) + ' Done')
                splits.append((i, path.join(self.folder, split_fn)))
            else:
                break
        print('All splited successfully')
        return splits