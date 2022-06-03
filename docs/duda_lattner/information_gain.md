Dear PhD. Stefan Lattner,

We are getting in contact to inqure whether we could ask you for details on the
mathematical formulations used in the paper "Imposing Higher-Level Structure in
Polyphonic Music Generation using Convolutional Restricted Boltzmann Machines and Constraints (2018)".

My name is Martin Miguel, I am a PhD Candidate at the University of Buenos
Aires, tutoring graduate student Lucas Somacal who is working on style transfer
in symbolic music using embbeding transformations from a Variational Autoencoder.

We were interested in the formulation of Information Rate provided in your paper
as a proxy for musicality (in this case, self-similarity). We were hoping you
could provide use with further detail on how the calculations for information
rate were performed. Next, we state in further detail what we gathered from the
paper and what are the particular questions we have.

In the paper you describe the output of the RBM as $\mathbf{v} \in \mathcal{R}^{TxP}$ 
with $T$ the length in time frames of the input and $P$ the pitch number.
Considering the MIDI input, we asume it to be $\mathbf{v} \in {0, 1}^{TxP}$.
We also consider a dataset of K songs $\mathbf{v}^k$ ($k \in [1..K]$),
with $\mathbf{v}_{tp}^k = 1 \text{ if pitch p is sounding at time t for song k}$.


With this set, you define Information Gain for a song $v_{0..N}^k$, being the
underscript the time frame as:

$$IR(v_{0..N}) = \frac{1}{N}\sum_n^N H(v_n) - H(v_n | v_{0..n-1})$$

And mention that $H(v_n)$ is obtained \emph{counting identical time slices} and
$H(v_n | v_{0..n-1})$ is obtained \emph{using a first-order Markov Chain}.
In any case, considering that the definition of entropy $H$ as a sum of
probabilities, \textbf{our main question} is how you defined $p(v_n)$ and $p(v_n |
v_{0..n-1})$.




