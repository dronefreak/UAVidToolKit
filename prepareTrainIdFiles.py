"""Convert UAVid RGB colour label images to single-channel train-ID images.

Walks the source directory tree looking for sequence subdirectories
(``seq<N>/Labels/``), applies :class:`~colorTransformer.UAVidColorTransformer`
to convert each 3-channel RGB label to a 1-channel integer train-ID image, and
saves the result under an equivalent ``TrainId/`` subtree in the target
directory.

Usage
-----
::

    python prepareTrainIdFiles.py -s <source_dir> -t <target_dir>

The source directory must follow the UAVid layout::

    <source_dir>/
        seq<N>/
            Labels/
                <frame>.png   ← 3-channel RGB label

Outputs are written to::

    <target_dir>/
        seq<N>/
            TrainId/
                <frame>.png   ← 1-channel uint8 train-ID label
"""

import os
import os.path as osp
import numpy as np
import argparse
from tqdm import tqdm
from colorTransformer import UAVidColorTransformer
from PIL import Image

#: Recognised image file extensions for label files.
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

clrEnc = UAVidColorTransformer()


def prepareTrainIDForDir(gtDirPath, saveDirPath):
	"""Convert all RGB label images in *gtDirPath* to train-ID images.

	Iterates over every sequence folder (``seq*``) found in *gtDirPath*,
	reads each label image from ``seq<N>/Labels/``, converts it to a
	single-channel train-ID image using
	:class:`~colorTransformer.UAVidColorTransformer`, and saves the result
	to the corresponding ``seq<N>/TrainId/`` folder inside *saveDirPath*.
	The output directory is created automatically if it does not exist.

	Args:
	    gtDirPath (str): Root directory containing ground-truth sequence
	        folders (each with a ``Labels/`` subdirectory).
	    saveDirPath (str): Root directory where ``TrainId/`` output folders
	        will be created.  May be the same as *gtDirPath*.

	Raises:
	    AssertionError: If the output directory cannot be created.
	"""
	gt_paths = sorted([p for p in os.listdir(gtDirPath) if p.startswith('seq')])
	for pd in tqdm(gt_paths, desc='Sequences'):
		lbl_dir = osp.join(gtDirPath, pd, 'Labels')
		lbl_paths = sorted(
			[
				p
				for p in os.listdir(lbl_dir)
				if osp.splitext(p)[1].lower() in IMAGE_EXTENSIONS
			]
		)
		out_dir = osp.join(saveDirPath, pd, 'TrainId')
		if not osp.isdir(out_dir):
			os.makedirs(out_dir)
			assert osp.isdir(out_dir), 'Failed to create directory: %s' % out_dir
		for lbl_p in lbl_paths:
			lbl_path = osp.abspath(osp.join(lbl_dir, lbl_p))
			trainId_path = osp.join(out_dir, lbl_p)
			gt = np.array(Image.open(lbl_path))
			trainId = clrEnc.transform(gt, dtype=np.uint8)
			Image.fromarray(trainId).save(trainId_path)


def parseArgs():
	"""Parse command-line arguments.

	Returns:
	    argparse.Namespace: Parsed arguments with attributes:
	        ``source_dir`` and ``target_dir``.
	"""
	parser = argparse.ArgumentParser(
		description='Convert UAVid RGB label images to single-channel train-ID images.'
	)
	parser.add_argument(
		'-s',
		dest='source_dir',
		help='Source directory containing seq*/Labels/ subdirectories.',
	)
	parser.add_argument(
		'-t',
		dest='target_dir',
		help='Target directory where seq*/TrainId/ folders are written.',
	)
	return parser.parse_args()


if __name__ == '__main__':
	args = parseArgs()
	prepareTrainIDForDir(args.source_dir, args.target_dir)
