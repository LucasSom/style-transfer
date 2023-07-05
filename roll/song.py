import os.path
from typing import List

import model.colab_tension_vae.params as params
from model.colab_tension_vae.preprocess_midi import preprocess_midi_wrapper
from roll.guoroll import GuoRoll


class Song:
    def __init__(self, midi_file: str, nombre: str, audio_path: str, save_midi=True, pulso="negra",
                 granularity="semicorchea", verbose=False):
        self.name = nombre
        self.bars = params.config.bars
        self.pulso = pulso
        self.granularity = granularity

        matrices, _, old_pm, bars_skipped = preprocess_midi_wrapper(midi_file, verbose=verbose)
        self.old_pm = old_pm if save_midi else None
        self.bars_skipped = bars_skipped
        self.rolls = [
            GuoRoll(m, f"{nombre}_{i}", os.path.join(audio_path, f"{self.bars}bars"), song=self, save_midi=save_midi,
                    verbose=verbose)
            for i, m in enumerate(matrices)
        ]

    def get_adjacent_intervals(self, voice='melody') -> List[int]:
        return [i for r in self.rolls for i in r.get_adjacent_intervals(voice)]

    def get_adjacent_rhythmic_patterns(self, voice='melody') -> List[str]:
        return [i for r in self.rolls for i in r.get_adjacent_rhythmic_patterns(voice)]
