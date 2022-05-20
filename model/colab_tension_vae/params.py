class VAEConfigBase:
    def __init__(self, bars=None):
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

            self.z_dim = 96  # espacio latente

            self.TEMPO = 90
            self.melody_dim = self.melody_output_dim
            self.bass_dim = self.bass_output_dim
            self.velocity = 100

            self.SLIDING_WINDOW = self.SEGMENT_BAR_LENGTH


config_name = ''
config = VAEConfigBase()
configs = {
    '1bar': VAEConfigBase(1),
    '4bar': VAEConfigBase(4),
    '8bar': VAEConfigBase(8),
}


def init(_config_name="8bar"):
    global config
    global config_name

    if isinstance(_config_name, int):
        _config_name = f"{_config_name}bar"

    config_name = _config_name
    config = configs[config_name]
