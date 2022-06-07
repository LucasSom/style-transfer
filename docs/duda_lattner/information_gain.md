Dear PhD. Stefan Lattner,

We are getting in contact to inqure whether we could ask you for details on the
mathematical formulations used in the paper "Imposing Higher-Level Structure in
Polyphonic Music Generation using Convolutional Restricted Boltzmann Machines
and Constraints (2018)".

My name is Martin Miguel, I am a PhD Candidate at the University of Buenos
Aires, tutoring graduate student Lucas Somacal who is working on style transfer
in symbolic music using embbeding transformations as obtained from a
Variational Autoencoder.

We were interested in the formulation of Information Rate provided in your paper
as a proxy for musicality (in this case, self-similarity). We were hoping you
could provide further detail on how the calculations for Information
Rate were performed. Next, we state what we gathered from the
paper and what are the particular questions we have.

In the paper you describe the output of the RBM as $\mathbf{v} \in \mathcal{R}^{T\times P}$ 
with $T$ the length in time frames of the input and $P$ the pitch number.
Considering a MIDI input, we asume it to be $\mathbf{v} \in \{0, 1\}^{T\times P}$.
We also consider a dataset of $K$ songs $\mathbf{v}^k$ ($k \in [1..K]$),
with $\mathbf{v}_{tp}^k = 1$ if pitch $p$ is sounding at time $t$ for song $k$.

With these definitions, you define Information Gain for a song $v_{0..N}^k$ (being the
underscript the time frame) as:

$$IR(v_{0..N}) = \frac{1}{N}\sum\limits_n^N H(v_n) - H(v_n | v_{0..n-1})$$

And mention that $H(v_n)$ is obtained \emph{counting identical time slices} and
$H(v_n | v_{0..n-1})$ is obtained \emph{using a first-order Markov Chain}.
Here, considering that the definition of entropy $H$ as a sum of
probabilities, \textbf{our main questions} are what is meant by $v_{0..N}$
and how you defined $p(v_n)$ and $p(v_n | v_{0..n-1})$.

Regarding $v_{0..N}$, we understand each $v_i$ indicates the musical events
sounding at time $i$. We understand this can mean one of two things:

* $v_i$ is a column vector $\{0, 1\}^P$ indicating which pitches sound at time
  $i$
* there's an asumption of monophony and $v_i$ is a number $p$ indicating the
  pitch that is soundng (or 0 if theres no sound).

Regarding $p(v_n)$, we are in doubt if it refers to the number of times any
song $k$ takes on value $v_n$ at time $n$ or ar any possible time frame. That
is:

* $p(v_n) = \frac{1}{K} \sum_{k=0}^K \delta(v_n^k = v_n)$
* $p(v_n) = \frac{1}{TK} \sum_{k=t}^T \sum_{k=0}^K \delta(v_{nt}^k = v_n)$

Finally, regarding $p(v_n | v_{0..n-1})$, we understand that the Markov
property indicates we asume $p(v_n | v_{0..n-1}) = p(v_n | v_{n-1})$. Then we
would define it as:

* $p(v_n | v_{n-1}) = \frac{\sum_{k=t}^{T-1} \sum_{k=0}^K \delta(v_{nt}^k = v_n, v_{(n-1)t}^k = v_{n-1})}{
                            \sum_{k=t}^{T-1} \sum_{k=0}^K \delta(v_{(n-1)t}^k = v_{n-1})}$
We greatly appreciate the time taken to read this and thank you for any further
clarification that can be provided.

Kind regards,
Martin Miguel and Lucas Somacal
