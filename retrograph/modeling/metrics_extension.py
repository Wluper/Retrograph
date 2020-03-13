import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import sets
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import distribution_strategy_context
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export

def mcc(labels,
              predictions,
              weights=None,
              metrics_collections=None,
              updates_collections=None,
              name=None):

  if context.executing_eagerly():
    raise RuntimeError('mcc is not '
                       'supported when eager execution is enabled.')

  with tf.variable_scope(name, 'mcc',(predictions, labels, weights)):

    predictions, labels, weights = _remove_squeezable_dimensions(
        predictions=math_ops.cast(predictions, dtype=dtypes.bool),
        labels=math_ops.cast(labels, dtype=dtypes.bool),
        weights=weights)

    true_p, true_positives_update_op = tf.metrics.true_positives(
        labels,
        predictions,
        weights,
        metrics_collections=None,
        updates_collections=None,
        name=None)
    false_p, false_positives_update_op = tf.metrics.false_positives(
        labels,
        predictions,
        weights,
        metrics_collections=None,
        updates_collections=None,
        name=None)
    true_n, true_negatives_update_op = tf.metrics.true_negatives(
        labels,
        predictions,
        weights,
        metrics_collections=None,
        updates_collections=None,
        name=None)
    false_n, false_negatives_update_op = tf.metrics.false_negatives(
        labels,
        predictions,
        weights,
        metrics_collections=None,
        updates_collections=None,
        name=None)

    def compute_mcc(tp, fp, tn, fn, name):
      return tf.math.divide(
          tf.math.subtract(
            tf.math.multiply(tp,tn),
            tf.math.multiply(fp,fn))
          ,tf.sqrt(
            tf.math.multiply(
              tf.math.multiply(tf.math.add(tp,fp),tf.math.add(tp,fn)),
              tf.math.multiply(tf.math.add(tn,fp),tf.math.add(tn,fn)))), name=name)

    def once_across_towers(_, true_p, false_p, true_n, false_n):
      return compute_mcc(true_p, false_p, true_n, false_n, 'value')

    mcc = _aggregate_across_towers(metrics_collections, once_across_towers,
                                 true_p, false_p, true_n, false_n)

    update_op = compute_mcc(true_positives_update_op,
                                  false_positives_update_op, true_negatives_update_op, false_negatives_update_op, 'update_op')
    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return mcc, update_op


def _remove_squeezable_dimensions(predictions, labels, weights):
  """Squeeze or expand last dim if needed.

  Squeezes last dim of `predictions` or `labels` if their rank differs by 1
  (using confusion_matrix.remove_squeezable_dimensions).
  Squeezes or expands last dim of `weights` if its rank differs by 1 from the
  new rank of `predictions`.

  If `weights` is scalar, it is kept scalar.

  This will use static shape if available. Otherwise, it will add graph
  operations, which could result in a performance hit.

  Args:
    predictions: Predicted values, a `Tensor` of arbitrary dimensions.
    labels: Optional label `Tensor` whose dimensions match `predictions`.
    weights: Optional weight scalar or `Tensor` whose dimensions match
      `predictions`.

  Returns:
    Tuple of `predictions`, `labels` and `weights`. Each of them possibly has
    the last dimension squeezed, `weights` could be extended by one dimension.
  """
  predictions = ops.convert_to_tensor(predictions)
  if labels is not None:
    labels, predictions = confusion_matrix.remove_squeezable_dimensions(
        labels, predictions)
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())

  if weights is None:
    return predictions, labels, None

  weights = ops.convert_to_tensor(weights)
  weights_shape = weights.get_shape()
  weights_rank = weights_shape.ndims
  if weights_rank == 0:
    return predictions, labels, weights

  predictions_shape = predictions.get_shape()
  predictions_rank = predictions_shape.ndims
  if (predictions_rank is not None) and (weights_rank is not None):
    # Use static rank.
    if weights_rank - predictions_rank == 1:
      weights = array_ops.squeeze(weights, [-1])
    elif predictions_rank - weights_rank == 1:
      weights = array_ops.expand_dims(weights, [-1])
  else:
    # Use dynamic rank.
    weights_rank_tensor = array_ops.rank(weights)
    rank_diff = weights_rank_tensor - array_ops.rank(predictions)

    def _maybe_expand_weights():
      return control_flow_ops.cond(
          math_ops.equal(rank_diff, -1),
          lambda: array_ops.expand_dims(weights, [-1]), lambda: weights)

    # Don't attempt squeeze if it will fail based on static check.
    if ((weights_rank is not None) and
        (not weights_shape.dims[-1].is_compatible_with(1))):
      maybe_squeeze_weights = lambda: weights
    else:
      maybe_squeeze_weights = lambda: array_ops.squeeze(weights, [-1])

    def _maybe_adjust_weights():
      return control_flow_ops.cond(
          math_ops.equal(rank_diff, 1), maybe_squeeze_weights,
          _maybe_expand_weights)

    # If weights are scalar, do nothing. Otherwise, try to add or remove a
    # dimension to match predictions.
    weights = control_flow_ops.cond(
        math_ops.equal(weights_rank_tensor, 0), lambda: weights,
        _maybe_adjust_weights)
  return predictions, labels, weights

def _aggregate_across_towers(metrics_collections, metric_value_fn, *args):
  """Aggregate metric value across towers."""
  def fn(distribution, *a):
    """Call `metric_value_fn` in the correct control flow context."""
    if hasattr(distribution, '_outer_control_flow_context'):
      # If there was an outer context captured before this method was called,
      # then we enter that context to create the metric value op. If the
      # caputred context is `None`, ops.control_dependencies(None) gives the
      # desired behavior. Else we use `Enter` and `Exit` to enter and exit the
      # captured context.
      # This special handling is needed because sometimes the metric is created
      # inside a while_loop (and perhaps a TPU rewrite context). But we don't
      # want the value op to be evaluated every step or on the TPU. So we
      # create it outside so that it can be evaluated at the end on the host,
      # once the update ops have been evaluted.

      # pylint: disable=protected-access
      if distribution._outer_control_flow_context is None:
        with ops.control_dependencies(None):
          metric_value = metric_value_fn(distribution, *a)
      else:
        distribution._outer_control_flow_context.Enter()
        metric_value = metric_value_fn(distribution, *a)
        distribution._outer_control_flow_context.Exit()
        # pylint: enable=protected-access
    else:
      metric_value = metric_value_fn(distribution, *a)
    if metrics_collections:
      ops.add_to_collections(metrics_collections, metric_value)
    return metric_value

  return distribution_strategy_context.get_tower_context().merge_call(fn, *args)
