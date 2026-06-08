"""Blend UAVid images with their colour-coded label overlays.

For each frame in a dataset split, this script alpha-composites the original
camera image with its corresponding colour label image (or train-ID label,
which is automatically converted to RGB) using OpenCV's
:func:`cv2.addWeighted`.  The result is saved as a new image in an output
directory, making it easy to visually inspect segmentation quality.

Usage
-----
::

    python blendImageAndLabel.py \\
        -i <image_dir> -l <label_dir> -o <output_dir> \\
        [-id <image_subdir>] [-ld <label_subdir>] [-od <output_subdir>] \\
        [-alpha <float>] [-beta <float>] [-gamma <float>]

Default blend formula: ``output = image * 0.6 + label * 0.4 + 0.0``

Note
----
Both images are loaded via Pillow (RGB channel order).  ``cv2.addWeighted``
performs a channel-wise weighted sum and does not reorder channels, so the
RGB channel order is preserved throughout.
"""

import os
import os.path as osp
import numpy as np
import argparse
import cv2
from PIL import Image
from tqdm import tqdm
from colorTransformer import UAVidColorTransformer

clrEnc = UAVidColorTransformer()


def blendImageAndLabelForDir(
	imgDirHome,
	labelDirHome,
	blendDirHome,
	imageSubDirname='Images',
	labelSubDirname='Labels',
	blendSubDirname='Blend',
	alpha=0.6,
	beta=0.4,
	gamma=0.0,
):
	"""Blend all image/label pairs in a directory tree and save the results.

	Iterates over every sequence folder (``seq*``) in *labelDirHome*, loads
	the matching camera image and label image for each frame, and writes the
	blended result to *blendDirHome*.  Single-channel train-ID label images
	are automatically converted to RGB before blending.

	The blend is computed as::

	    output[i] = image[i] * alpha + label[i] * beta + gamma

	Args:
	    imgDirHome (str): Root directory of camera images
	        (``seq<N>/<imageSubDirname>/``).
	    labelDirHome (str): Root directory of label images
	        (``seq<N>/<labelSubDirname>/``).
	    blendDirHome (str): Root directory where blended outputs are saved
	        (``seq<N>/<blendSubDirname>/``).
	    imageSubDirname (str, optional): Subdirectory name for camera images.
	        Defaults to ``'Images'``.
	    labelSubDirname (str, optional): Subdirectory name for label images.
	        Defaults to ``'Labels'``.
	    blendSubDirname (str, optional): Subdirectory name for blend outputs.
	        Defaults to ``'Blend'``.
	    alpha (float, optional): Weight applied to the camera image channel
	        values.  Defaults to ``0.6``.
	    beta (float, optional): Weight applied to the label image channel
	        values.  Defaults to ``0.4``.
	    gamma (float, optional): Scalar added to every output pixel after
	        the weighted sum.  Defaults to ``0.0``.

	Raises:
	    AssertionError: If the output directory cannot be created.
	    ValueError: If the camera image and label image have different spatial
	        dimensions (H×W).
	"""
	lbl_seq_paths = sorted([p for p in os.listdir(labelDirHome) if p.startswith('seq')])
	for pd in tqdm(lbl_seq_paths, desc='Sequences'):
		lbl_dir = osp.join(labelDirHome, pd, labelSubDirname)
		lbl_paths = sorted(os.listdir(lbl_dir))
		bld_dir = osp.join(blendDirHome, pd, blendSubDirname)
		if not osp.isdir(bld_dir):
			os.makedirs(bld_dir)
			assert osp.isdir(bld_dir), 'Cannot create directory: %s' % bld_dir
		for lbl_p in lbl_paths:
			lbl_path = osp.abspath(osp.join(lbl_dir, lbl_p))
			img_path = osp.join(imgDirHome, pd, imageSubDirname, lbl_p)
			bld_path = osp.join(bld_dir, lbl_p)
			lbl = np.array(Image.open(lbl_path))
			img = np.array(Image.open(img_path))
			# If the label is a single-channel train-ID image, convert to RGB
			# before blending so both arrays have the same number of channels.
			if lbl.ndim == 2:
				lbl = clrEnc.inverse_transform(lbl)
			if img.shape[:2] != lbl.shape[:2]:
				raise ValueError(
					'Image and label size mismatch for %s: '
					'img=%s, label=%s' % (lbl_p, img.shape[:2], lbl.shape[:2])
				)
			blendImg = cv2.addWeighted(img, alpha, lbl, beta, gamma)
			Image.fromarray(blendImg).save(bld_path)


def parseArgs():
	"""Parse command-line arguments.

	Returns:
	    argparse.Namespace: Parsed arguments with attributes:
	        ``image_dir``, ``label_dir``, ``output_dir``,
	        ``image_subdir``, ``label_subdir``, ``output_subdir``,
	        ``alpha``, ``beta``, and ``gamma``.
	"""
	parser = argparse.ArgumentParser(
		description='Blend UAVid camera images with colour-coded label overlays.'
	)
	parser.add_argument(
		'-i', dest='image_dir', type=str, help='Root directory of camera images.'
	)
	parser.add_argument(
		'-l', dest='label_dir', type=str, help='Root directory of label images.'
	)
	parser.add_argument(
		'-o',
		dest='output_dir',
		type=str,
		help='Root directory for blended output images.',
	)
	parser.add_argument(
		'-id',
		dest='image_subdir',
		type=str,
		default='Images',
		help='Image subdirectory name inside each sequence folder (default: Images).',
	)
	parser.add_argument(
		'-ld',
		dest='label_subdir',
		type=str,
		default='Labels',
		help='Label subdirectory name inside each sequence folder (default: Labels).',
	)
	parser.add_argument(
		'-od',
		dest='output_subdir',
		type=str,
		default='Blend',
		help='Output subdirectory name inside each sequence folder (default: Blend).',
	)
	parser.add_argument(
		'-alpha',
		type=float,
		default=0.6,
		help='Weight for the camera image (default: 0.6).',
	)
	parser.add_argument(
		'-beta',
		type=float,
		default=0.4,
		help='Weight for the label image (default: 0.4).',
	)
	parser.add_argument(
		'-gamma',
		type=float,
		default=0.0,
		help='Scalar added to every output pixel (default: 0.0).',
	)
	return parser.parse_args()


if __name__ == '__main__':
	args = parseArgs()
	blendImageAndLabelForDir(
		args.image_dir,
		args.label_dir,
		args.output_dir,
		args.image_subdir,
		args.label_subdir,
		args.output_subdir,
		args.alpha,
		args.beta,
		args.gamma,
	)
