# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: all,-jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.10.6
# ---

# %% tags=[]
# %matplotlib ipympl

# %% tags=[]
from pathlib import Path

import blackjax
import corner
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pyhf

# %% tags=[]
# Create figures directory from file name
figure_dir = Path(str(Path(__file__)).rsplit(".")[0] + "_figures")
figure_dir.mkdir(parents=True, exist_ok=True)

# %% tags=[]
pyhf.set_backend("jax")

model = pyhf.simplemodels.uncorrelated_background([5, 15, 10], [50, 50, 50], [5, 7, 4])
obs_data = pyhf.tensorlib.astensor([50, 75, 65] + model.config.auxdata)

logprob_func = jax.jit(lambda pars: model.logpdf(pars, obs_data)[0])
init_pars = jnp.array(model.config.suggested_init())

# %% tags=[]
hmc = blackjax.hmc(
    logprob_func,
    step_size=1e-3,
    inverse_mass_matrix=jnp.eye(len(init_pars)),
    num_integration_steps=1000,
)


# %% tags=[]
def inference_loop(rng_key, kernel, initial_state, num_samples):
    @jax.jit
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states


# %% tags=[]
rng_key = jax.random.PRNGKey(0)
init_state = hmc.init(init_pars)
_, rng_key = jax.random.split(rng_key)
states = inference_loop(rng_key, hmc.step, init_state, 50_000)

# %% tags=[]
burn_in = 3000

fig, ax = plt.subplots(1, len(init_pars), figsize=(12, 2))
for i, (axi, name) in enumerate(zip(ax, model.config.par_names)):
    axi.plot(states.position[:, i])
    axi.set_title(f"{name}")
    axi.axvline(x=burn_in, color="tab:red")

extensions = ["png", "pdf", "svg"]
for ext in extensions:
    fig.savefig(figure_dir / f"chains.{ext}")

# %% tags=[]
chains = states.position[burn_in:, :]
n_samples, _ = chains.shape

# %% tags=[]
fig = corner.corner(np.array(chains), labels=model.config.par_names)

for ext in extensions:
    fig.savefig(figure_dir / f"corner.{ext}")

# %%
almost_posterior_predictive = jax.vmap(model.expected_data)(chains[:300])
print(f"{almost_posterior_predictive.shape=}")

# %% tags=[]
fig, ax = plt.subplots()

ax.step(
    np.arange(len(obs_data)),
    almost_posterior_predictive.T,
    alpha=0.02,
    where="mid",
    color="steelblue",
)
ax.scatter(np.arange(len(obs_data)), obs_data, color="black")
ax.axvline(2.5, color="red")
ax.set_title("real data | aux data")

for ext in extensions:
    fig.savefig(figure_dir / f"predictions.{ext}")
