# This is TF2.0 port of dpu_utils.tfutils.unsortedsegmentops.
# Should be migrated back into a dpu_utils.tf2utils package.

import tensorflow as tf

SMALL_NUMBER = 1e-7


def unsorted_segment_logsumexp(scores, segment_ids, num_segments):
    """Perform an unsorted segment safe logsumexp."""
    # Note: if a segment is empty, the smallest value for the score will be returned,
    # which yields the correct behavior
    max_per_segment = tf.math.unsorted_segment_max(
        data=scores, segment_ids=segment_ids, num_segments=num_segments
    )
    scattered_log_maxes = tf.gather(params=max_per_segment, indices=segment_ids)
    recentered_scores = scores - scattered_log_maxes
    exped_recentered_scores = tf.math.exp(recentered_scores)

    per_segment_sums = tf.math.unsorted_segment_sum(
        exped_recentered_scores, segment_ids, num_segments
    )
    per_segment_logs = tf.math.log(per_segment_sums)
    return per_segment_logs + max_per_segment


def unsorted_segment_log_softmax(logits, segment_ids, num_segments):
    """Perform an unsorted segment safe log_softmax."""
    # Note: if a segment is empty, the smallest value for the score will be returned,
    # which yields the correct behavior
    max_per_segment = tf.math.unsorted_segment_max(
        data=logits, segment_ids=segment_ids, num_segments=num_segments
    )
    scattered_maxes = tf.gather(params=max_per_segment, indices=segment_ids)
    recentered_scores = logits - scattered_maxes
    exped_recentered_scores = tf.math.exp(recentered_scores)

    per_segment_sums = tf.math.unsorted_segment_sum(
        exped_recentered_scores, segment_ids, num_segments
    )
    per_segment_normalization_consts = tf.math.log(per_segment_sums)

    log_probs = recentered_scores - tf.gather(
        params=per_segment_normalization_consts, indices=segment_ids
    )
    return log_probs


def unsorted_segment_softmax(logits, segment_ids, num_segments):
    """Perform a safe unsorted segment softmax."""
    max_per_segment = tf.math.unsorted_segment_max(
        data=logits, segment_ids=segment_ids, num_segments=num_segments
    )
    scattered_maxes = tf.gather(params=max_per_segment, indices=segment_ids)
    recentered_scores = logits - scattered_maxes
    exped_recentered_scores = tf.math.exp(recentered_scores)

    per_segment_sums = tf.math.unsorted_segment_sum(
        exped_recentered_scores, segment_ids, num_segments
    )

    probs = exped_recentered_scores / (
        tf.gather(params=per_segment_sums, indices=segment_ids) + SMALL_NUMBER
    )
    return probs
