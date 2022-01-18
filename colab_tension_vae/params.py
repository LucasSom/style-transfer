rnn_dim = 256
input_dim = 89

time_step = 64 * 2	# 64 = 4 compases

start_middle_dim = 64 * 2
melody_bass_dense_1_dim = 128

melody_output_dim = 74
melody_note_start_dim = 1
bass_output_dim = 13
bass_note_start_dim = 1
# 89

tension_middle_dim = 128
tension_output_dim = 1

z_dim = 96 # espacio latente

TEMPO = 90
melody_dim = melody_output_dim
bass_dim = bass_output_dim
velocity = 100

SAMPLES_PER_BAR = 16
SEGMENT_BAR_LENGTH = 4 * 2 	# 4*4 compases
SLIDING_WINDOW=SEGMENT_BAR_LENGTH

