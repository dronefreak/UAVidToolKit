"""Color transformation utilities for the UAVid dataset.

This module provides :class:`UAVidColorTransformer`, which maps between the
UAVid dataset's 3-channel RGB label images and compact single-channel integer
label (train-ID) images, and vice-versa.

UAVid label colour mapping
--------------------------
+-------------+--------------------+
| Class name  | RGB colour         |
+=============+====================+
| Clutter     | (0,   0,   0)      |
| Building    | (128, 0,   0)      |
| Road        | (128, 64,  128)    |
| Static_Car  | (192, 0,   192)    |
| Tree        | (0,   128, 0)      |
| Vegetation  | (128, 128, 0)      |
| Human       | (64,  64,  0)      |
| Moving_Car  | (64,  0,   128)    |
+-------------+--------------------+

The train-ID assigned to each class equals its insertion order in the table
above (0 = Clutter, 1 = Building, …, 7 = Moving_Car).
"""

import numpy as np


class UAVidColorTransformer:
	"""Bidirectional converter between RGB colour labels and integer train-IDs.

	The converter encodes each RGB triple as a unique integer key using a
	base-256 hash so that every possible 24-bit colour maps to a distinct
	slot with no collisions.

	Attributes:
	    clr_tab (dict): Ordered mapping of class name → ``[R, G, B]`` list.
	    id_tab  (dict): Mapping of class name → scalar integer hash of its RGB
	        colour, used for O(1) pixel classification in :meth:`transform`.
	"""

	def __init__(self):
		"""Initialise the transformer by building the colour and hash tables."""
		self.clr_tab = self.createColorTable()
		# Pre-compute the integer hash for every colour so transform() avoids
		# recomputing it on every call.
		id_tab = {}
		for k, v in self.clr_tab.items():
			id_tab[k] = self.clr2id(v)
		self.id_tab = id_tab

	def createColorTable(self):
		"""Build and return the class-name → RGB colour mapping.

		The insertion order defines the integer train-ID for each class.

		Returns:
		    dict: Ordered mapping of ``{class_name: [R, G, B]}``.
		"""
		clr_tab = {}
		clr_tab['Clutter'] = [0, 0, 0]
		clr_tab['Building'] = [128, 0, 0]
		clr_tab['Road'] = [128, 64, 128]
		clr_tab['Static_Car'] = [192, 0, 192]
		clr_tab['Tree'] = [0, 128, 0]
		clr_tab['Vegetation'] = [128, 128, 0]
		clr_tab['Human'] = [64, 64, 0]
		clr_tab['Moving_Car'] = [64, 0, 128]
		return clr_tab

	def colorTable(self):
		"""Return the class-name → RGB colour mapping.

		Returns:
		    dict: Reference to the internal ``{class_name: [R, G, B]}`` dict.
		"""
		return self.clr_tab

	def clr2id(self, clr):
		"""Compute a collision-free integer hash for an RGB triple.

		Each channel occupies exactly 256 slots (values 0–255), giving a
		unique 24-bit integer for every possible RGB colour.

		Args:
		    clr (list[int]): ``[R, G, B]`` colour with values in ``[0, 255]``.

		Returns:
		    int: Unique integer hash ``R + G*256 + B*256*256``.
		"""
		return clr[0] + clr[1] * 256 + clr[2] * 256 * 256

	def transform(self, label, dtype=np.int32):
		"""Convert a 3-channel RGB label image to a single-channel train-ID image.

		Each pixel whose RGB value matches a known class colour is assigned
		that class's integer train-ID.  Pixels that do not match any known
		colour default to ``0`` (Clutter).

		Args:
		    label (numpy.ndarray): HxWx3 uint8 array containing RGB label
		        colours as loaded directly from a UAVid label image.
		    dtype (numpy.dtype, optional): Dtype of the output array.
		        Defaults to ``numpy.int32``.

		Returns:
		    numpy.ndarray: HxW array of integer train-IDs with the given
		    *dtype*.
		"""
		height, width = label.shape[:2]
		# Initialise all pixels to 0 (Clutter) — unknown colours remain Clutter.
		newLabel = np.zeros((height, width), dtype=dtype)
		# Promote to int64 before hashing to avoid uint8 overflow during
		# the multiplication (256*256 = 65536 > 255).
		id_label = label.astype(np.int64)
		id_label = (
			id_label[:, :, 0] + id_label[:, :, 1] * 256 + id_label[:, :, 2] * 256 * 256
		)
		for tid, key in enumerate(self.clr_tab.keys()):
			mask = id_label == self.id_tab[key]
			newLabel[mask] = tid
		return newLabel

	def inverse_transform(self, label):
		"""Convert a single-channel train-ID image back to a 3-channel RGB image.

		Args:
		    label (numpy.ndarray): HxW array of integer train-IDs, as
		        produced by :meth:`transform`.

		Returns:
		    numpy.ndarray: HxWx3 uint8 array with RGB colours matching the
		    UAVid colour table.
		"""
		label_img = np.zeros(shape=(label.shape[0], label.shape[1], 3), dtype=np.uint8)
		values = list(self.clr_tab.values())
		for tid, val in enumerate(values):
			label_img[label == tid] = val
		return label_img
