# cython: language_level=3
"""Cython wrapper around the C++ confusion-matrix accumulation routine.

This module exposes a single public function, :func:`cEvaluatePair`, which
delegates to a hand-written C implementation (``addToConfusionMatrix_impl.c``)
for high-throughput pixel-level confusion-matrix updates.

Build instructions
------------------
Compile from the repository root with::

    python setup.py build_ext --inplace

Only tested on Ubuntu 64-bit with g++.

Note
----
The ``@cython.boundscheck(False)`` decorator is applied to :func:`cEvaluatePair`
to eliminate per-element index-bounds checks, which is safe here because the
array sizes are validated before the C call.
"""

import numpy as np
cimport cython
cimport numpy as np
import ctypes

np.import_array()

# ---------------------------------------------------------------------------
# C function declaration
# ---------------------------------------------------------------------------

cdef extern from "addToConfusionMatrix_impl.c":
    void addToConfusionMatrix( const unsigned char* f_prediction_p  ,
                               const unsigned char* f_groundTruth_p ,
                               const unsigned int   f_width_i       ,
                               const unsigned int   f_height_i      ,
                               unsigned long long*  f_confMatrix_p  ,
                               const unsigned int   f_confMatDim_i  )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

cdef tonumpyarray(unsigned long long* data, unsigned long long size):
    """Wrap a raw C ``unsigned long long`` buffer as a 2-D NumPy uint64 array.

    The returned array shares memory with *data* — use ``np.copy()`` before
    the underlying buffer goes out of scope.

    Args:
        data: Pointer to the first element of a ``size × size`` flat buffer.
        size: Side length of the square confusion matrix.

    Returns:
        numpy.ndarray: ``size × size`` view into *data* with dtype uint64.

    Raises:
        ValueError: If *data* is NULL or *size* is negative.
    """
    if not (data and size >= 0):
        raise ValueError('data pointer is NULL or size is negative.')
    return np.PyArray_SimpleNewFromData(
        2, [size, size], np.NPY_UINT64, <void*>data
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@cython.boundscheck(False)
def cEvaluatePair( np.ndarray[np.uint8_t , ndim=2] predictionArr  ,
                   np.ndarray[np.uint8_t , ndim=2] groundTruthArr ,
                   np.ndarray[np.uint64_t, ndim=2] confMatrix     ,
                   evalLabels                                      ):
    """Add one prediction/ground-truth image pair to the confusion matrix.

    Converts input arrays to C-contiguous memory layout (if needed) and
    delegates to the C++ implementation for efficient pixel-level accumulation.
    The confusion matrix is updated in-place inside the C routine and a copy
    is returned.

    Args:
        predictionArr (numpy.ndarray): HxW uint8 array of predicted train-IDs.
        groundTruthArr (numpy.ndarray): HxW uint8 array of GT train-IDs.
            Must have the same shape as *predictionArr*.
        confMatrix (numpy.ndarray): Square uint64 confusion matrix that is
            accumulated over multiple calls.  Mutated in place by the C
            routine; a defensive copy is returned.
        evalLabels: Sequence of class IDs included in the evaluation (unused
            by the C routine, kept for API compatibility with the Python path).

    Returns:
        numpy.ndarray: Updated copy of *confMatrix* (uint64, same shape).
    """
    cdef np.ndarray[np.uint8_t    , ndim=2, mode="c"] predictionArr_c
    cdef np.ndarray[np.uint8_t    , ndim=2, mode="c"] groundTruthArr_c
    cdef np.ndarray[np.ulonglong_t, ndim=2, mode="c"] confMatrix_c

    # Ensure C-contiguous layout required by the C routine.
    predictionArr_c  = np.ascontiguousarray(predictionArr , dtype=np.uint8    )
    groundTruthArr_c = np.ascontiguousarray(groundTruthArr, dtype=np.uint8    )
    confMatrix_c     = np.ascontiguousarray(confMatrix    , dtype=np.ulonglong)

    # numpy shape: axis-0 = rows = height, axis-1 = columns = width.
    cdef np.uint32_t height_ui     = predictionArr.shape[0]
    cdef np.uint32_t width_ui      = predictionArr.shape[1]
    cdef np.uint32_t confMatDim_ui = confMatrix.shape[0]

    addToConfusionMatrix(
        &predictionArr_c[0, 0], &groundTruthArr_c[0, 0],
        width_ui, height_ui,
        &confMatrix_c[0, 0], confMatDim_ui
    )

    # Wrap the raw buffer as a NumPy array and return a copy so that the
    # caller owns the data independently of confMatrix_c's lifetime.
    confMatrix = np.ascontiguousarray(
        tonumpyarray(&confMatrix_c[0, 0], confMatDim_ui)
    )
    return np.copy(confMatrix)
