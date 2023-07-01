import roll.guoroll
import pandas as pd
d = pd.read_pickle('./data/brmf_4b/Evaluation/cross_val/df_80_indexes_1.pkl')
d
d = pd.read_pickle('./data/brmf_4b/Evaluation/cross_val/rolls_long_df_test_0.pkl')
d.columns
ls data
ls data/brfm_4b
ls data/brmf_4b
ls data/brmf_4b/embeddings
ls data/brmf_4b/
ls data/brmf_4b/vae
ls data/datasets
ls data/preprocessed_data/
d = pd.read_pickle('./data/preprocessed_data_path/bach-rag-moz-fres-4.pkl')
d
d.iloc[0]['roll']
r = d.iloc[0]['roll']
r.display_score()
r
r.score
r.score.plot('roll')
r.score.plot()
r.score.elements
r.score.elements[0]
r.score.elements[0].elements
r.score.elements[1].elements
import roll.guoroll
roll.guoroll.roll_permutations
ps = roll.guoroll.roll_permutations(r, 5)
roll.guoroll.roll_permutations??
roll.guoroll.params
roll.guoroll.params.config
roll.guoroll.params.config
help(roll.guoroll.params.config)
help(roll.guoroll.params.config)
roll.guoroll.params.config.melody_dim
import params
import model.colab_tension_vae.params
model.colab_tension_vae.params.init(4)
ps = roll.guoroll.roll_permutations(r, 5)
ps
ps[0]
len(ps)
ps.shape
ps[0] == ps[1]
ps.shape
r
r.get_melody()
r.get_melody() == ps[0]
r.get_melody().sum()
ps[0].sum()
ps[1].sum()
ps[2].sum()
ps[0].shape
r.get_melody().shape
ps[:74]
ps[0][:74]
ps[0][:74].shape
ps[0][:74, :].shape
ps[0][:, :74].shape
r.get_melody().shape
ps[0][:, :74].sum()
r.matrix
r.matrix.shape
fig, axs = plt.subplots(6, 1)
%pylab
fig, axs = plt.subplots(6, 1)
for m, ax in zip([r.matrix] + ps, axs):
    plt.sca(ax)
    plt.imshow(m, interpolate='closest')
plt.imshow?
for m, ax in zip([r.matrix] + ps, axs):
    plt.sca(ax)
    plt.imshow(m, interpolation='closest')
for m, ax in zip([r.matrix] + ps, axs):
    plt.sca(ax)
    plt.imshow(m, interpolation='nearest')
fig, axs = plt.subplots(6, 1)
for m, ax in zip([r.matrix] + ps, axs):
    plt.sca(ax)
    plt.imshow(m.T, interpolation='nearest')
fig, axs = plt.subplots(6, 1)
for m, ax in zip([r.matrix] + ps, axs):
    plt.sca(ax)
    plt.imshow(m.T, interpolation='nearest')
plt.tight_layout()
plt.tight_layout()
fig, axs = plt.subplots(6, 1, figsize=(20, 3))
for m, ax in zip([r.matrix] + ps, axs):
    plt.sca(ax)
    plt.imshow(m.T, interpolation='nearest')
fig, axs = plt.subplots(1,6, figsize=(20, 3))
for m, ax in zip([r.matrix] + ps, axs):
    plt.sca(ax)
    plt.imshow(m.T, interpolation='nearest')
fig, axs = plt.subplots(1,6, figsize=(20, 3))
for m, ax in zip([r.matrix] + ps, axs):
    plt.sca(ax)
    plt.imshow(m.T, interpolation='nearest')
import utils.audio_management
import util
import utils.utils
import model.colab_tension_vae.util as uitl
import model.colab_tension_vae.util as util
util
util.roll_to_pretty_midi?
util.roll_to_pretty_midi(ps)
util.roll_to_pretty_midi(ps, r.midi)
util.roll_to_pretty_midi(ps[0], r.midi)
r.midi
help(r)
import pretty_midi
r.midi
pretty_midi.PrettyMIDI(r.midi)
pretty_midi.PrettyMIDI(r.midi.replace('_0.mp3', ''))
r
r.name
ls data/preprocessed_data/
ls data/preprocessed_data/original/
ls data/preprocessed_data/original/
util.roll_to_pretty_midi(ps[0], None)
pm = util.roll_to_pretty_midi(ps[0], None)
pm.write?
import utils.audio_management
pm.write('0.mid')
utils.audio_management.PlayMidi('0.mid', '0.wav')
ps
r
ls
mkdir temp
cd temp
ls
for n, m in zip(['r', range(5)], [r] + ps):
    midi = util.roll_to_pretty_midi(m, None)
    midi.write(f'{n}.mid')
    utils.audio_management.PlayMidi(f'{n}.mid', f'{n}.wav')
for n, m in zip(['r', range(5)], [r.matrix] + ps):
    midi = util.roll_to_pretty_midi(m, None)
    midi.write(f'{n}.mid')
    utils.audio_management.PlayMidi(f'{n}.mid', f'{n}.wav')
for n, m in zip(['r'] + range(5), [r.matrix] + ps):
    midi = util.roll_to_pretty_midi(m, None)
    midi.write(f'{n}.mid')
    utils.audio_management.PlayMidi(f'{n}.mid', f'{n}.wav')
for n, m in zip(['r'] + list(range(5)), [r.matrix] + ps):
    midi = util.roll_to_pretty_midi(m, None)
    midi.write(f'{n}.mid')
    utils.audio_management.PlayMidi(f'{n}.mid', f'{n}.wav')
ms = [r.matrix] + ps
cat_m = []
import random
random.shuffle(ms)
cat_m = []
ms[0].shape
for m in ms:
    cat_m.append(m)
    cat_m.append(np.zeros(8, 89))
for m in ms:
    cat_m.append(m)
    cat_m.append(np.zeros((8, 89)))
cat_m
len(cat_m)
cat_m = []
for m in ms:
    cat_m.append(m)
    cat_m.append(np.zeros((8, 89)))
cat_m
np.concat(cat_m)
np.concat(cat_ms)
np.concatenate(cat_ms)
np.concatenate(cat_m)
np.concatenate(cat_m).shape
cat_m_silence np.concatenate(cat_m).shape
cat_m_silence = np.concatenate(cat_m).shape
cat_midi = util.roll_to_pretty_midi(cat_m_silence, None)
cat_m_silence.shape
cat_m_silence
cat_m_silence = np.concatenate(cat_m)
cat_midi = util.roll_to_pretty_midi(cat_m_silence, None)
cat_midi.write('cat.mid')
utils.audio_management.PlayMidi('cat.mid', 'cat.wav')
%history
%history -f permutation_exploration.py
