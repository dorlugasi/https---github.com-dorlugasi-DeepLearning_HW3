r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=64,
        seq_len=64,
        h_dim=256,
        n_layers=3,
        dropout=0.3,
        learn_rate=1e-3,
        lr_sched_factor=0.5,
        lr_sched_patience=2,
    )
    # ========================
    return hypers


def part1_generation_params():
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "ACT I.\nSCENE I.\n"
    temperature = 0.5
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**

we split the corpus into fixed-length sequences because training on the entire text at once is computationally infeasible and inefficient. Using sequences allows us to apply backpropagation through time on manageable chunks, fit the data into memory, and train in mini-batches. This also stabilizes training while still enabling the RNN to learn long-term dependencies through its hidden state.
"""

part1_q2 = r"""
**Your answer:**

even thought training is done on fixed-length sequences, the RNN maintains a hidden state that carries information forward across sequence boundaries. During training, the hidden state from the end of one sequence is passed as the initial state of the next, allowing the model to accumulate context over time. During generation, the hidden state is never reset, so the model can exhibit dependencies and structure that extend well beyond the sequence length used in training.
"""

part1_q3 = r"""
**Your answer:**

we do not shuffle the order of batches because the RNN relies on temporal continuity between sequences. The hidden state from one batch is used to initialize the next batch, preserving the correct order of the text. Shuffling the batches would break this temporal structure and prevent the model from learning meaningful long-term dependencies.
"""

part1_q4 = r"""
**Your answer:**

We lower the temperature when sampling in order to reduce randomness and generate more readable and consistent text. When the temperature is very high, the model samples characters more randomly, which often leads to meaningless or incoherent text. When the temperature is very low, the model almost always picks the most likely character, which makes the text repetitive and less creative.

"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=32, h_dim=1024, z_dim=64, x_sigma2=0.5, learn_rate=2e-4, betas=(0.5, 0.999),
    )
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**

The hyperparameter $\sigma^2$ (x_sigma2) controls how much noise the model assumes in the reconstruction of the input

On the one hand, a low value of $\sigma^2$ makes the reconstruction loss stronger, so the model focuses more on accurately reconstructing the input images.

On the other hand, a high value of $\sigma^2$ weakens the reconstruction loss, allowing the model to tolerate larger reconstruction errors and focus more on learning a smoother latent space.
"""

part2_q2 = r"""
**Your answer:**

the reconstruction loss encourages the VAE to accurately reconstruct the input data from the latent representation. The KL divergence loss regularizes the latent-space distribution by pushing it to be close to a standard normal distribution.

as a result, the latent space becomes smooth and well-structured, with nearby latent points corresponding to similar outputs. The benefit of this is that the model can sample meaningful and diverse new data points and generalize better.

"""

part2_q3 = r"""
**Your answer:**

we start by maximizing the evidence distribution $p(X)$ because it represents how likely the model is to generate the observed data.

maximizing $p(X)$ means learning model parameters that best explain the training data. 

since $p(X)$ is intractable to compute directly in VAEs, we optimize a lower bound instead, which leads to the VAE loss formulation.
"""

part2_q4 = r"""
**Your answer:**

we model the log of the latent-space variance $\log(\sigma_\alpha^2)$ instead of directly modeling $\sigma_\alpha^2$ in order to ensure that the variance is always positive.

predicting $\log(\sigma_\alpha^2)$ allows the network to output unconstrained real values, while the actual variance is obtained by exponentiation. 
This improves numerical stability and makes training more stable.

"""


def part3_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim = 0, 
        num_heads = 0,
        num_layers = 0,
        hidden_dim = 0,
        window_size = 0,
        droupout = 0.0,
        lr=0.0,
    )

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======
    hypers = dict(
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        hidden_dim = 128,
        window_size = 8,
        droupout = 0.1,
        lr = 5e-4,
    )
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**

Each sliding-window attention layer allows every token to attend only to a local neighborhood. when stacking multiple encoder layers, the information from neighboring tokens is gradually combined and passed to the next layer. as a result, tokens in higher layers indirectly receive information from tokens that are farther away. this is similar to stacking CNN layers, where deeper layers have a larger receptive field. 

Therefore, the final layer captures a broader context even though each individual attention layer is local.

"""

part3_q2 = r"""
**Your answer:**

One possible variation is to extend the sliding-window attention by adding a small number of long-range connections. in addition to attending to a local window, each token also attends to a few tokens at larger, fixed distances (for example, exponentially spaced positions).

Since the total number of attended tokens per position is still bounded by $w$, the computational complexity remains $O(nw)$. these long-range connections allow global information to be shared more efficiently, because distant tokens can influence each other directly instead of only through many local steps. as a result, fewer layers are required to capture global context compared to standard sliding-window attention.

However, the attention pattern is still sparse, so not all token pairs interact directly, which limits full global information sharing.

"""

# ==============
