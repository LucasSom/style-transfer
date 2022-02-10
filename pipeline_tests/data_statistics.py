import math

from preprocessing import preprocess_midi_wrapper


class SongStatistics:
    def __init__(self, file):
        roll, filled_indices, midi, bars_skipped = preprocess_midi_wrapper(file)

        if midi is not None:
            self.cambios_de_compas = len(midi.time_signature_changes) - 1
            self.tempo_estimado = midi.estimate_tempo()  # en bpm
            last_note_ending = max(midi.instruments[0].notes[-1].end, midi.instruments[1].notes[-1].end)
            self.cantidad_de_compases_estimada = math.ceil(last_note_ending / self.tempo_estimado)
            self.cantidad_de_rolls_estimados = \
                self.cantidad_de_compases_estimada / 8 + (self.cantidad_de_compases_estimada % 8 != 0)

            self.cantidad_de_rolls = len(filled_indices)
            self.cantidad_de_compases = self.cantidad_de_rolls * 8
        else:
            self.cambios_de_compas = None
            self.tempo_estimado = None
            self.cantidad_de_compases_estimada = None
            self.cantidad_de_rolls_estimados = None
            self.cantidad_de_rolls = None
            self.cantidad_de_compases = None

        self.compases_salteados = bars_skipped


    def print_stats(self):
        print("Cambios de compás: ", self.cambios_de_compas)
        print("Tempo estimado (en bpm): ", self.tempo_estimado)
        print("Cantidad de compases estimada: ", self.cantidad_de_compases_estimada)
        print("Cantidad de rolls estimados: ", self.cantidad_de_rolls_estimados)
        print("Cantidad de compases: ", self.cantidad_de_compases)
        print("Cantidad de rolls: ", self.cantidad_de_rolls)
        print("Cantidad de rolls salteados: ", len(self.compases_salteados))
        print("Compases salteados: ", self.compases_salteados)
