import os
from math import cos, pi, sin

from typing import Sequence
from functools import partial

from sklearn.model_selection import train_test_split

from scipy.integrate import odeint

import pandas as pd

import jax
import jax.numpy as jnp                # JAX NumPy

from flax import linen as nn           # The Linen API
from flax.training import train_state  # Useful dataclass to keep train state

import tensorflow as tf

import optax                           # Optimizers

from apng import APNG
from PIL import Image, ImageDraw

class MLP(nn.Module):

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(32)(x)
        x = nn.tanh(x)
        x = nn.Dense(1)(x)
        return x

@jax.jit
def MSE_loss(preds, targets):
    return jnp.square(preds - targets).mean()

def fvec(x, t, state):
    def sum_hamiltonian(params, x):
        hamiltonian = state.apply_fn({'params': state.params}, x)
        return jnp.sum(hamiltonian)

    dhdu_dhdv = jax.jit(jax.grad(sum_hamiltonian, argnums=1))

    gradient = dhdu_dhdv(state.params, x)
    return jnp.matmul(gradient, St)

@partial(jax.jit, static_argnums=(3,))
def step(inputs, targets, state, is_train=True):

    def sum_hamiltonian(params, x):
        hamiltonian = state.apply_fn({'params': params}, x)
        return jnp.sum(hamiltonian)

    dhdu_dhdv = jax.jit(jax.grad(sum_hamiltonian, argnums=1))

    def grad(params, x):
        gradient = dhdu_dhdv(params, x)
        return jnp.matmul(gradient, St)

    def loss_fn(params):
        preds = grad(params, inputs)
        loss = MSE_loss(preds, targets)
        return loss, preds

    grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

    if is_train:
        (loss, preds), grads = grad_fn(state.params)
        state = state.apply_gradients(grads = grads)
    else:
        loss, preds = loss_fn(state.params)

    return loss, preds, state

def make_animation(index, qval, pval):
    filename = "{:0>4}.png".format(index)
    im = Image.new("RGB", (100, 100), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    x = 30*sin(qval) + 50
    y = 30*cos(qval) + 50
    draw.line((50, 50, x, y), fill=(0, 255, 0), width=2)
    draw.ellipse((x-5, y-5, x+5, y+5), fill=(0, 0, 255))
    im.save(filename)
    return filename

if __name__ == "__main__":
    MY_BATCH_SIZE = 100

    dftarget = pd.read_csv("target.csv", header=None, dtype=jnp.float32)
    dfinput = pd.read_csv("input.csv", header=None, dtype=jnp.float32)

    X_train, X_test, Y_train, Y_test = train_test_split(dfinput.values, dftarget.values, test_size=0.2)

    train_ds = tf.data.Dataset.from_tensors((X_train, Y_train))
    train_ds = train_ds.shuffle(len(train_ds), seed=0, reshuffle_each_iteration=True).batch(MY_BATCH_SIZE).prefetch(1)

    test_ds = tf.data.Dataset.from_tensors((X_test, Y_test))
    test_ds = test_ds.batch(MY_BATCH_SIZE).prefetch(1)

    N = 1
    O = jnp.zeros((N,N))
    Id = jnp.eye(N)
    S = jnp.vstack([jnp.hstack([O, Id]), jnp.hstack([-Id, O])])
    St = S.T

    rng = jax.random.PRNGKey(0)
    learning_rate = 0.001

    model = MLP()
    params = model.init(rng, jnp.ones(2))['params'] # initialize parameters by passing a template image
    tx = optax.adam(learning_rate)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    num_epochs = 3000

    for epoch in range(num_epochs):
        train_loss, eval_loss = 0.0, 0.0
        for x, y in train_ds.as_numpy_iterator():
            loss, preds, state = step(x, y, state, is_train=True)
            train_loss = train_loss + loss

        for x, y in test_ds.as_numpy_iterator():
            loss, preds, state = step(x, y, state, is_train=False)
            eval_loss = eval_loss + loss

        print("{}/{} training loss: {}, evaluation loss: {}".format(epoch+1, num_epochs, train_loss, eval_loss))

    x0 = jnp.asarray([1.0, 0.0])
    teval = jnp.linspace(0.0, 10.0, 100)
    args = (state,)

    orbit = odeint(fvec, x0, teval, args)

    os.makedirs("./visualization/animation/", exist_ok=True)

    os.chdir("./visualization/")

    files = []
    for i in range(100):
        files.append(make_animation(i, orbit[i,0], orbit[i,1]))
    APNG.from_files(files, delay=50).save("./animation/animation_model.png")