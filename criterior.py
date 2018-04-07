import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import linalg_ops

tfgan = tf.contrib.gan


MODEL_GRAPH_DEF = 'classify_mnist_graph_def.pb'
INPUT_TENSOR = 'inputs:0'
OUTPUT_TENSOR = 'logits:0'

def _graph_def_from_par_or_disk(filename):
    if filename is None:
      return tfgan.eval.get_graph_def_from_resource(MODEL_GRAPH_DEF)
    else:
      return tfgan.eval.get_graph_def_from_disk(filename)


def _kl_divergence(p, p_logits, q):

    for tensor in [p, p_logits, q]:
          if not tensor.dtype.is_floating:
              raise ValueError('Input %s must be floating type.', tensor.name)
    p.shape.assert_has_rank(2)
    p_logits.shape.assert_has_rank(2)
    q.shape.assert_has_rank(1)
    return math_ops.reduce_sum(
        p * (nn_ops.log_softmax(p_logits) - math_ops.log(q)), axis=1)

def _kl_divergence_without_logits(p,q):
    for tensor in [p, q]:
        if not tensor.dtype.is_floating:
            raise ValueError('Input %s must be floating type.', tensor.name)
    return math_ops.reduce_sum(
        p * (math_ops.log(p) - math_ops.log(q)), axis=1)

def classifier_score(images, classifier_fn, num_batches=1):

    generated_images_list = array_ops.split(
        images, num_or_size_splits=num_batches)

    # Compute the classifier splits using the memory-efficient `map_fn`.
    logits = functional_ops.map_fn(
        fn=classifier_fn,
        elems=array_ops.stack(generated_images_list),
        parallel_iterations=1,
        back_prop=False,
        swap_memory=True,
        name='RunClassifier')
    logits = array_ops.concat(array_ops.unstack(logits), 0)

    return classifier_score_from_logits(logits)

def classifier_score_from_logits(logits):

    logits.shape.assert_has_rank(2)

    # Use maximum precision for best results.
    logits_dtype = logits.dtype
    if logits_dtype != dtypes.float64:
      logits = math_ops.to_double(logits)

    p = nn_ops.softmax(logits)
    q = math_ops.reduce_mean(p, axis=0)

    kl = _kl_divergence(p, logits, q)
    kl.shape.assert_has_rank(1)
    log_score = math_ops.reduce_mean(kl)
    final_score = math_ops.exp(log_score)

    if logits_dtype != dtypes.float64:
      final_score = math_ops.cast(final_score, logits_dtype)

    return final_score


def mnist_score(images, graph_def_filename=None, input_tensor=INPUT_TENSOR,
                output_tensor=OUTPUT_TENSOR, num_batches=32):

    graph_def = _graph_def_from_par_or_disk(graph_def_filename)
    mnist_classifier_fn = lambda x: tfgan.eval.run_image_classifier(  # pylint: disable=g-long-lambda
        x, graph_def, input_tensor, output_tensor)

    # score = tfgan.eval.classifier_score(
       # tf.reshape(images,[-1,28,28,1]), mnist_classifier_fn, num_batches)

    score = classifier_score(
       tf.reshape(images,[-1,28,28,1]), mnist_classifier_fn, num_batches)
    score.shape.assert_is_compatible_with([])

    return score

def mnist_score_new(images, graph_def_filename=None, input_tensor=INPUT_TENSOR,
                output_tensor=OUTPUT_TENSOR, num_batches=32):

    graph_def = _graph_def_from_par_or_disk(graph_def_filename)
    mnist_classifier_fn = lambda x: tfgan.eval.run_image_classifier(  # pylint: disable=g-long-lambda
        x, graph_def, input_tensor, output_tensor)

    score = classifier_score_new(
        tf.reshape(images,[-1,28,28,1]), mnist_classifier_fn, num_batches)
    score.shape.assert_is_compatible_with([])

    return score

def classifier_score_new(images, classifier_fn, num_batches=1):

    images_list = array_ops.split(
        images, num_or_size_splits=num_batches)

    # Compute the classifier splits using the memory-efficient `map_fn`.

    logits = functional_ops.map_fn(
        fn=classifier_fn,
        elems=array_ops.stack(images_list),
        parallel_iterations=1,
        back_prop=False,
        swap_memory=True,
        name='RunClassifier')
    logits = array_ops.concat(array_ops.unstack(logits), 0)
    return classifier_score_from_logits_new(logits)

def classifier_score_from_logits_new(logits):

    logits.shape.assert_has_rank(2)

    # Use maximum precision for best results.
    logits_dtype = logits.dtype
    if logits_dtype != dtypes.float64:
      logits = math_ops.to_double(logits)

    p = nn_ops.softmax(logits)
    q = math_ops.reduce_mean(p, axis=0)
    kl = _kl_divergence_without_logits(p, (p+q)/2) + _kl_divergence_without_logits(q, (p+q)/2)
    kl.shape.assert_has_rank(1)
    log_score = math_ops.reduce_mean(kl)
    final_score = math_ops.exp(log_score)

    if logits_dtype != dtypes.float64:
      final_score = math_ops.cast(final_score, logits_dtype)

    return final_score


def mnist_frechet_distance(real_images, generated_images,
                           graph_def_filename=None, input_tensor=INPUT_TENSOR,
                           output_tensor=OUTPUT_TENSOR, num_batches=32):
   # """Frechet distance between real and generated images."""
    graph_def = _graph_def_from_par_or_disk(graph_def_filename)
    mnist_classifier_fn = lambda x: tfgan.eval.run_image_classifier(
        x, graph_def, input_tensor, output_tensor)

    frechet_distance = tfgan.eval.frechet_classifier_distance(
      tf.reshape(real_images,[-1,28,28,1]), tf.reshape(generated_images,[-1,28,28,1]), mnist_classifier_fn, num_batches)
    return frechet_distance

def frechet_classifier_distance_new(real_images,
                                generated_images,
                                classifier_fn,
                                num_batches=1):

    real_images_list = array_ops.split(
      real_images, num_or_size_splits=num_batches)
    generated_images_list = array_ops.split(
      generated_images, num_or_size_splits=num_batches)

    imgs = array_ops.stack(real_images_list + generated_images_list)

    # Compute the activations using the memory-efficient `map_fn`.
    activations = functional_ops.map_fn(
      fn=classifier_fn,
      elems=imgs,
      parallel_iterations=1,
      back_prop=False,
      swap_memory=True,
      name='RunClassifier')

  # Split the activations by the real and generated images.
    real_a, gen_a = array_ops.split(activations, [num_batches, num_batches], 0)

  # Ensure the activations have the right shapes.
    real_a = array_ops.concat(array_ops.unstack(real_a), 0)
    gen_a = array_ops.concat(array_ops.unstack(gen_a), 0)

    return frechet_classifier_distance_from_activations_new(real_a, gen_a)

def frechet_classifier_distance_from_activations_new(
    real_activations, generated_activations):

    real_activations.shape.assert_has_rank(2)
    generated_activations.shape.assert_has_rank(2)

    activations_dtype = real_activations.dtype
    if activations_dtype != dtypes.float64:
      real_activations = math_ops.to_double(real_activations)
      generated_activations = math_ops.to_double(generated_activations)

    # Compute mean and covariance matrices of activations.
    m = math_ops.reduce_mean(real_activations, 0)
    m_v = math_ops.reduce_mean(generated_activations, 0)
    num_examples = math_ops.to_double(array_ops.shape(real_activations)[0])

    # sigma = (1 / (n - 1)) * (X - mu) (X - mu)^T
    real_centered = real_activations - m
    sigma = math_ops.matmul(
      real_centered, real_centered, transpose_a=True) / (num_examples - 1)

    gen_centered = generated_activations - m_v
    sigma_v = math_ops.matmul(
      gen_centered, gen_centered, transpose_a=True) / (num_examples - 1)

    # Find the Tr(sqrt(sigma sigma_v)) component of FID
    sqrt_trace_component = math_ops.trace(math_ops.sqrt(sigma) * sigma_v * math_ops.sqrt(sigma))

    # Compute the two components of FID.

    # First the covariance component.
    # Here, note that trace(A + B) = trace(A) + trace(B)
    trace = math_ops.trace(sigma + sigma_v) - 2.0 * sqrt_trace_component

    # Next the distance between means.
    mean = math_ops.square(linalg_ops.norm(m - m_v))  # This uses the L2 norm.
    fid = trace + mean
    if activations_dtype != dtypes.float64:
      fid = math_ops.cast(fid, activations_dtype)

    return fid


def mnist_frechet_distance_new(real_images, generated_images,
                           graph_def_filename=None, input_tensor=INPUT_TENSOR,
                           output_tensor=OUTPUT_TENSOR, num_batches=1):
  # """Frechet distance between real and generated images.
    graph_def = _graph_def_from_par_or_disk(graph_def_filename)
    mnist_classifier_fn = lambda x: tfgan.eval.run_image_classifier(
        x, graph_def, input_tensor, output_tensor)

    frechet_distance = frechet_classifier_distance_new(
      tf.reshape(real_images,[-1,28,28,1]), tf.reshape(generated_images,[-1,28,28,1]), mnist_classifier_fn, num_batches)
    return frechet_distance


