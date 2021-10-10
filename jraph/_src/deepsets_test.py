import jax
import jax.numpy as jnp
import haiku as hk

import jax.tree_util as tree
import jraph
from jraph._src import graph
from jraph._src import models
from jraph._src import utils

# define little models.DeepSets model

# this block squeezes nodes to update the global attributes
@jraph.concatenated_args
def deepset_node_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
  """Node update deepset function for graph net."""

  net = hk.Sequential(
      [
       PermEquil_mean(128), # deepset layer 1
       jax.nn.elu,
       PermEquil_mean(128),
       jax.nn.elu,
       PermEquil_mean(128),
       jax.nn.elu
      ]
  )

  return net(feats)


@jraph.concatenated_args
def deepset_update_global_fn(feats: jnp.ndarray) -> jnp.ndarray:
  """Global update function for graph net."""
  # we want to sum-pool all our encoded nodes
  #feats = feats.sum(axis=-1) # sum-pool
  net = hk.Sequential(
      [hk.Linear(128), jax.nn.elu,
       hk.Linear(30), jax.nn.elu,
       hk.Linear(11)]) # number of variabilities
  return net(feats)



def net_fn(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
  """Graph net function."""
  # Add a global paramater for graph classification.
  graph = graph._replace(globals=jjnp.zeros([graph.n_node.shape[0], 1]))
  embedder = jraph.GraphMapFeatures(
      hk.Linear(128), hk.Linear(128), hk.Linear(128))
  net = jraph.GraphNetwork(
      update_node_fn=deepset_node_update_fn,
      update_edge_fn=edge_update_fn,
      update_global_fn=update_global_fn)
  return net(embedder(graph))


# define network instance

mynet = models.DeepSets(update_node_fn=deepset_node_update_fn,
                         update_global_fn=deepset_update_global_fn)


mynet = hk.without_apply_rng(hk.transform(mynet))
# get a single Deepset graph
# some node data
node_dat = jnp.stack([jnp.sin(jnp.arange(50)*0.5), jnp.exp(-jnp.arange(50)*0.5)], axis=-1)

single_graph = jraph.GraphsTuple(
    n_node=jnp.asarray([len(node_dat)]), n_edge=jnp.asarray([0]),
    nodes=node_dat, edges=None,
    globals=jnp.ones((1, 2)),
    senders=None, receivers=None)


params = mynet.init(jax.random.PRNGKey(42), single_graph)

print(params['linear']['w'])

print('yay we can now apply a DeepSet to a graph !')
