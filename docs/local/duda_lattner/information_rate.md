Dear PhD. Stefan Lattner,

We are getting in contact to inquire whether we could ask you for details on the
mathematical formulations used in the paper "Imposing Higher-Level Structure in
Polyphonic Music Generation using Convolutional Restricted Boltzmann Machines
and Constraints (2018)".

My name is Martin Miguel, I am a PhD Candidate at the University of Buenos
Aires, tutoring graduate student Lucas Somacal who is working on style transfer
in symbolic music using embedding transformations as obtained from a
Variational Autoencoder.

We were interested in the formulation of Information Rate provided in your paper
as a proxy for musicality (in this case, self-similarity). We were hoping you
could provide further detail on how the calculations for Information
Rate were performed. Next, we state what we gathered from the
paper and what are the particular questions we have.

In the paper you describe the output of the RBM as $\mathbf{v} \in \mathbb{R}^{T\times P}$ 
with $T$ the length in time frames of the input and $P$ the pitch number.
Considering a MIDI input, we assume it to be $\mathbf{v} \in \{0, 1\}^{T\times P}$.
We also consider a dataset of $K$ songs $\mathbf{v}^k$ ($k \in [1..K]$),
with $\mathbf{v}_{tp}^k = 1$ if pitch $p$ is sounding at time $t$ for song $k$.

With these definitions, you define Information Gain for a song $v_{0..N}^k$ (being the
underscript the time frame) as:

$$IR(v_{0..N}) = \frac{1}{N}\sum_{n=1}^N H(v_n) - H(v_n | v_{0..n-1})$$

And mention that $H(v_n)$ is obtained _counting identical time slices_ and
$H(v_n | v_{0..n-1})$ is obtained _using a first-order Markov Chain_.
Here, considering the definition of entropy $H$ as a sum of
probabilities, *our main questions* are what is meant by $v_n$
and how you defined $p(v_n)$ and $p(v_n | v_{0..n-1})$.

Regarding $v_n$, we understand it indicates the musical events
sounding at time $n$. We understand this can mean one of two things:

* $v_n$ is a column vector $\{0, 1\}^P$ indicating which pitches sound at time
  $n$
* there's an assumption of monophony and $v_n$ is a number $p$ indicating the
  pitch that is sounding (or 0 if there's no sound).

Regarding $p(v_n)$, we are in doubt if it refers to the number of times any
song $k$ takes on value $v_n$ at time $n$ or at any possible time frame. That
is:

* $p(v_n) = \frac{1}{K} \sum_{k=0}^K \delta(\mathbf{v}_n^k = v_n)$\qquad or
* $p(v_n) = \frac{1}{TK} \sum_{k=t}^T \sum_{k=0}^K \delta(\mathbf{v}_{nt}^k = v_n)$

Finally, regarding $p(v_n | v_{0..n-1})$, we understand that the Markov
property indicates that we assume $p(v_n | v_{0..n-1}) = p(v_n | v_{n-1})$. Then we
would define it as:

* $p(v_n | v_{n-1}) = \frac{\sum_{t=1}^{T} \sum_{k=0}^K \delta(\mathbf{v}_{t}^k = v_n, \mathbf{v}_{(t-1)}^k = v_{n-1})}{
                            \sum_{t=1}^{T} \sum_{k=0}^K \delta(\mathbf{v}_{(t-1)}^k = v_{n-1})}$


We greatly appreciate the time taken to read this and thank you for any further
clarification that can be provided.


Kind regards,

Martin Miguel and Lucas Somacal
