import tensorflow_probability as tfp 
import tensorflow as tf 
td=tfp.distributions
initial_distrubation=tfd.Categorical(probs=[0.8,0.2])
transition_distrubation=tfd.Categorical(probs=[[0.7,0.3],[0.2,0.8]])
observation_distrubtion=tfd.Normal(loc=[0.,15.],scale=[5.,10.])
model=tfd.HiddenMarkovModel(
    initial_distrubation=initial_distrubation,
    transition_distrubation=transition_distrubation,
    observation_distrubtion=observation_distrubtion,
    num_steps=7,
)
mean = model.mean()
with tf.compat.v1.Session() as sess: 
  print(mean.numpy())