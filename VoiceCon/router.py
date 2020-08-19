from datetime import datetime
from os import path
from pathlib import Path
import json

from main.lib import Cloner
from utils.audio_chunk import SplitWavAudioMubin


class Router:

    def __init__(self, config):
        self.CONFIG = config
        self.cloner = Cloner(self.CONFIG['ENCODER_MODEL_PATH'], self.CONFIG['SYNTHESIZER_MODEL_PATH'], self.CONFIG['VOCODER_MODEL_PATH'])

    def audio2embeddings(self, audio, speaker, seq_lengths):
        audio_dir = path.join(self.CONFIG["DATASET_ROOT"], speaker)
        Path(audio_dir).mkdir(parents=True, exist_ok=True)
        filename = (
            speaker
            + "_"
            + datetime.now().strftime("%Y_%m_%d-%H_%M_%s")
            + ".%s" % audio.filename.split(".")[-1]
        )
        audio.save(path.join(audio_dir, filename))
        print(filename)

        obj = SplitWavAudioMubin(audio_dir, filename)
        seq_filenames = obj.multiple_split(seq_lengths)

        if not seq_filenames:
            seq_filenames = [filename]

        embeds_meta = self.cloner.synthesize_embeds(
            seq_filenames,
            speaker,
            Path(self.CONFIG["SYNTHESIZER_ROOT"]),
            Path(self.CONFIG["ENCODER_MODEL_PATH"]),
            self.CONFIG["N_PROCESS"],
            self.CONFIG["EMBED_CONFIG_PATH"],
        )
        if embeds_meta:
            status = True
        else:
            status = False
            embeds_meta = "Couldn't create embeddings..."

        return embeds_meta, status


    def text2speech(self, embed_id, texts):
        if not embed_id:
            return "Embed Id cannot be empty"
        if not texts:
            return "Please provide text to synthesize the audio"

        with open(self.CONFIG["EMBED_CONFIG_PATH"], "r") as handle:
            embeddings_metadata = json.load(handle)
            embed = embeddings_metadata.get(embed_id)
            if not embed:
                return "Invalid embedding..."

        save_path = path.join(self.CONFIG['STATIC_DIR'], embed['speaker'])
        Path(save_path).mkdir(parents=True, exist_ok=True)
        save_path = path.join(save_path, embed['speaker']+"-"+datetime.now().strftime("%Y_%m_%d-%H_%M_%s")+".wav")

        if type(texts) == list:
            texts = " ".join(texts)

        texts = texts.replace("\n", " ").split()
        texts = [" ".join(texts[i: i + 20]).replace(".", ". ").replace(",", ", ") for i in range(0, len(texts), 20)]
        print(texts)

        status = self.cloner.generate_audio(embed['embed_path'], texts, self.CONFIG["N_PROCESS"], save_path)

        if status:
            return self.CONFIG['API_URL']+save_path, status
        return "Couldn't generate audio", status