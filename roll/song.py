from model.colab_tension_vae.preprocess_midi import preprocess_midi_wrapper
from roll.roll import Roll


class Song:
    def __init__(self, midi_file: str, nombre: str, compases=8, pulso="negra", granularity="semicorchea"):
        self.nombre = nombre
        self.compases = compases
        self.pulso = pulso
        self.granularity = granularity

        matrices, _, old_pm, bars_skipped = preprocess_midi_wrapper(midi_file)
        self.old_pm = old_pm
        self.bars_skipped = bars_skipped
        self.rolls = [Roll(m, song=self, compases=compases) for m in matrices]
