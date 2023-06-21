class VAEConfigBase:
    def __init__(self, bars=None, z=96):
        if bars is not None:
            self.bars = bars

            self.SAMPLES_PER_BAR = 16
            self.SEGMENT_BAR_LENGTH = bars  # 4*4 compases

            self.rnn_dim = 256
            self.input_dim = 89

            self.time_step = bars * self.SAMPLES_PER_BAR  # 64 = 4 compases

            self.start_middle_dim = self.time_step
            self.melody_bass_dense_1_dim = 128

            self.melody_output_dim = 74
            self.melody_note_start_dim = 1
            self.bass_output_dim = 13
            self.bass_note_start_dim = 1
            # 89

            self.tension_middle_dim = 128
            self.tension_output_dim = 1

            self.z_dim = z  # espacio latente

            self.TEMPO = 90
            self.melody_dim = self.melody_output_dim
            self.bass_dim = self.bass_output_dim
            self.velocity = 100

            self.SLIDING_WINDOW = self.SEGMENT_BAR_LENGTH


config_name = ''
config = VAEConfigBase()
configs = {
    '1bar_z96': VAEConfigBase(1),
    '4bar_z96': VAEConfigBase(4),
    '8bar_z96': VAEConfigBase(8),
    '1bar_z20': VAEConfigBase(1, 20),
    '4bar_z20': VAEConfigBase(4, 20),
    '8bar_z20': VAEConfigBase(8, 20),
    '1bar_z192': VAEConfigBase(1, 192),
    '4bar_z192': VAEConfigBase(4, 192),
    '8bar_z192': VAEConfigBase(8, 192),
}


def init(bars, z=96):
    global config
    global config_name

    if isinstance(bars, int) or len(bars) == 1:
        bars = f"{bars}bar_z{z}"

    config_name = bars
    config = configs[config_name]
