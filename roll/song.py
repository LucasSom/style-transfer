import model.colab_tension_vae.params as params
from model.colab_tension_vae.preprocess_midi import preprocess_midi_wrapper
from roll.guoroll import GuoRoll


class Song:
    def __init__(self, midi_file: str, nombre: str, pulso="negra", granularity="semicorchea", verbose=False):
        self.nombre = nombre
        self.compases = params.config.bars
        self.pulso = pulso
        self.granularity = granularity

        matrices, _, old_pm, bars_skipped = preprocess_midi_wrapper(midi_file, verbose=verbose)
        self.old_pm = old_pm
        self.bars_skipped = bars_skipped
        self.rolls = [
            GuoRoll(m, song=self, verbose=verbose)
            for m in matrices
        ]
