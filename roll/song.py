from preprocessing import preprocess_midi_wrapper
from roll.roll import Roll


class Song:
    def __init__(self, midi_file, compases=4, pulso="negra", granularity="semicorchea"):
        self.compases = compases
        self.pulso = pulso
        self.granularity = granularity

        rolls, _, old_pm, bars_skipped = preprocess_midi_wrapper(midi_file)
        self.rolls = [Roll(r, song=self, compases=compases) for r in rolls]
        self.old_pm = old_pm
        self.bars_skipped = bars_skipped
