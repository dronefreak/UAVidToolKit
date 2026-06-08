"""Convert single-channel train-ID label images back to 3-channel RGB images.

Walks a directory tree containing train-ID label images (``seq<N>/TrainId/``),
applies the inverse colour transform from
:class:`~colorTransformer.UAVidColorTransformer`, and saves the resulting RGB
colour images into a configurable output subdirectory.

Usage
-----
::

    python convertTrainIdFiles2Color.py -s <source_dir> -t <target_dir> [-f <subdir>]

The source directory must follow the UAVid train-ID layout::

    <source_dir>/
        seq<N>/
            TrainId/
                <frame>.png   ← 1-channel uint8 train-ID label

Outputs are written to::

    <target_dir>/
        seq<N>/
            <subdir>/         ← defaults to 'color'
                <frame>.png   ← 3-channel RGB colour label
"""

import os
import os.path as osp
import numpy as np
import argparse
from tqdm import tqdm
from colorTransformer import UAVidColorTransformer
from PIL import Image

#: Recognised image file extensions for train-ID label files.
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

clrEnc = UAVidColorTransformer()


def convertTrainID2ColorForDir(trainIdDir, saveDirPath, subdirname='color'):
	"""Convert all train-ID images in *trainIdDir* to RGB colour images.

	Iterates over every sequence folder (``seq*``) found in *trainIdDir*,
	reads each train-ID image from ``seq<N>/TrainId/``, converts it to a
	3-channel RGB label image using
	:meth:`~colorTransformer.UAVidColorTransformer.inverse_transform`, and
	saves the result to ``seq<N>/<subdirname>/`` inside *saveDirPath*.
	The output directory is created automatically if it does not exist.

	Args:
	    trainIdDir (str): Root directory containing sequence folders with
	        ``TrainId/`` subdirectories of single-channel label images.
	    saveDirPath (str): Root directory where colour output folders are
	        written.  May be the same as *trainIdDir*.
	    subdirname (str, optional): Name of the output subdirectory created
	        inside each sequence folder.  Defaults to ``'color'``.

	Raises:
	    AssertionError: If the output directory cannot be created.
	"""
	trainId_paths = sorted([p for p in os.listdir(trainIdDir) if p.startswith('seq')])
	for pd in tqdm(trainId_paths, desc='Sequences'):
		lbl_dir = osp.join(trainIdDir, pd, 'TrainId')
		lbl_paths = sorted(
			[
				p
				for p in os.listdir(lbl_dir)
				if osp.splitext(p)[1].lower() in IMAGE_EXTENSIONS
			]
		)
		out_dir = osp.join(saveDirPath, pd, subdirname)
		if not osp.isdir(out_dir):
			os.makedirs(out_dir)
			assert osp.isdir(out_dir), 'Failed to create directory: %s' % out_dir
		for lbl_p in lbl_paths:
			lbl_path = osp.abspath(osp.join(lbl_dir, lbl_p))
			color_path = osp.join(out_dir, lbl_p)
			trainId = np.array(Image.open(lbl_path))
			colorImg = clrEnc.inverse_transform(trainId)
			Image.fromarray(colorImg).save(color_path)


def parseArgs():
	"""Parse command-line arguments.

	Returns:
	    argparse.Namespace: Parsed arguments with attributes:
	        ``source_dir``, ``target_dir``, and ``subdirname``.
	"""
	parser = argparse.ArgumentParser(
		description='Convert UAVid train-ID label images to RGB colour images.'
	)
	parser.add_argument(
		'-s',
		dest='source_dir',
		type=str,
		help='Source directory containing seq*/TrainId/ subdirectories.',
	)
	parser.add_argument(
		'-t',
		dest='target_dir',
		type=str,
		help='Target directory where colour output folders are written.',
	)
	parser.add_argument(
		'-f',
		dest='subdirname',
		type=str,
		default='color',
		help='Output subdirectory name inside each sequence folder (default: color).',
	)
	return parser.parse_args()


if __name__ == '__main__':
	args = parseArgs()
	convertTrainID2ColorForDir(args.source_dir, args.target_dir, args.subdirname)
