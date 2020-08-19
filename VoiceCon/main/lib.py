from pathlib import Path
from uuid import uuid4
import json
import numpy as np
from pathlib import Path
import librosa

from encoder import inference as encoder
from vocoder import inference as vocoder
from synthesizer.preprocess import create_embeddings
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder


class Cloner:

    def __init__(self, encoder_model_path, synthesizer_model_path, vocoder_model_path):
        print("Preparing the encoder, the synthesizer and the vocoder...")
        self.encoder = encoder
        # self.encoder.load_model(Path(encoder_model_path))
        self.synthesizer = Synthesizer(Path(synthesizer_model_path).joinpath("taco_pretrained"), low_mem=False)
        self.vocoder = vocoder
        self.vocoder.load_model(Path(vocoder_model_path))

    def synthesize_embeds(self, audio_seq_path, speaker, synthesizer_root, encoder_model_path, n_processes, embed_config_path):

        # Preprocess the dataset
        with open(embed_config_path, "r") as handle:
            embeddings_metadata = json.load(handle)

        embeddings = create_embeddings(audio_seq_path, speaker, synthesizer_root, encoder_model_path, n_processes)
        meta = []

        if embeddings:
            with open(embed_config_path, "w") as handle:
                for embed in embeddings:
                    em_id = str(uuid4())
                    embeddings_metadata[em_id] = {'audio_path': str(embed[0]), 'embed_path': str(embed[1]),'seq_length': embed[2], 'speaker': speaker}
                    meta.append({'embed_id': em_id, 'seq_length': embed[2]})

                json.dump(embeddings_metadata, handle)

        return meta

    def generate_audio(self, embed_path, texts, n_process, save_path):
        # preprocessed_wav = self.encoder.preprocess_wav(embed_path)
        # embed = encoder.embed_utterance(preprocessed_wav)
        # print(embed.shape)
        embed = np.load(embed_path)
        # text_len = ((len(texts) / n_process) * n_process) + (len(texts) % n_process) + n_process - (len(texts) % n_process)

        generated_wav = None
        for i in range(0, len(texts), n_process):
            embeds = np.stack([embed] * len(texts[i: i+n_process]))
            print(texts[i: i+n_process], embeds.shape)
            specs = self.synthesizer.synthesize_spectrograms(texts[i: i+n_process], embeds)
            breaks = [spec.shape[1] for spec in specs]
            print("breaks: ", breaks)
            spec = np.concatenate(specs, axis=1)

            wav = self.vocoder.infer_waveform(spec)
            # Add breaks
            b_ends = np.cumsum(np.array(breaks) * Synthesizer.hparams.hop_size)
            print("b_ends: ", b_ends)
            b_starts = np.concatenate(([0], b_ends[:-1]))
            print("b_starts: ", b_starts)
            wavs = [wav[start:end] for start, end, in zip(b_starts, b_ends)]
            breaks = [np.zeros(int(0.15 * Synthesizer.sample_rate))] * len(breaks)
            print("final breaks: ", breaks)
            wav = np.concatenate([i for w, b in zip(wavs, breaks) for i in (w, b)])

            if generated_wav is None:
                generated_wav = wav
            else:
                generated_wav = np.concatenate((generated_wav, wav))

        del embed
        del embeds
        del wav
        del wavs

        if generated_wav is not None:
            # Save it on the disk
            print(generated_wav.dtype)
            librosa.output.write_wav(save_path, generated_wav.astype(np.float32),
                                        self.synthesizer.sample_rate)
            print("\nSaved output as %s\n\n" % save_path)

            del generated_wav

            return True

        return False
