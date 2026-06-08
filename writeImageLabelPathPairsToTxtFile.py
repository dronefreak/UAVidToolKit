"""Write image–label path pairs for UAVid dataset splits to text files.

Each output text file contains one ``<image_path> <label_path>`` pair per
line.  These files are a convenient input format for many deep-learning
training frameworks.

Two independent modes are supported:

* **Train/val mode** (default): writes absolute paths for training and/or
  validation image–label pairs to ``./img_lbl_pair.txt``.
* **Test/pred mode** (``--test`` / ``--pred-dir``): writes image paths paired
  with expected prediction output paths for the test set (and optionally the
  validation set) to ``./test_pred_pair.txt`` and/or
  ``./valid_pred_pair.txt``.

Usage
-----
::

    # Training + validation ground-truth pairs
    python writeImageLabelPathPairsToTxtFile.py -d <dataset_dir> -t -v

    # Test image → expected prediction path pairs
    python writeImageLabelPathPairsToTxtFile.py -d <dataset_dir> --test --pred-dir <pred_dir>

Expected dataset layout::

    <dataset_dir>/
        train/  seq<N>/  Images/ + Labels/
        valid/  seq<N>/  Images/ + Labels/
        test/   seq<N>/  Images/
"""

import os
import os.path as osp
import argparse


def writeTrainValImageLabelPathPairsToTxtFile(
	data_home='./', useTrain=True, useVal=False
):
	"""Write image–label path pairs for the training and/or validation splits.

	Scans the ``train/`` and ``valid/`` subdirectories of *data_home*, pairs
	each image in ``Images/`` with its corresponding label in ``Labels/``, and
	writes all pairs (one per line, space-separated) to
	``./img_lbl_pair.txt``.

	Args:
	    data_home (str, optional): Root dataset directory containing ``train/``
	        and ``valid/`` subdirectories.  Defaults to ``'./'``.
	    useTrain (bool, optional): Include the training split.
	        Defaults to ``True``.
	    useVal (bool, optional): Include the validation split.
	        Defaults to ``False``.

	Raises:
	    AssertionError: If neither *useTrain* nor *useVal* is ``True``, or if
	        any expected image or label file does not exist.
	"""
	assert useTrain or useVal, 'Error: at least one of useTrain or useVal must be True.'
	train_home = osp.join(data_home, 'train')
	val_home = osp.join(data_home, 'valid')
	train_paths = sorted(os.listdir(train_home))
	val_paths = sorted(os.listdir(val_home))

	all_img_path = []
	all_lbl_path = []

	if useTrain:
		for pd in train_paths:
			img_dir = osp.join(train_home, pd, 'Images')
			img_paths = sorted(os.listdir(img_dir))
			for img_p in img_paths:
				img_path = osp.abspath(osp.join(img_dir, img_p))
				label_path = osp.abspath(osp.join(train_home, pd, 'Labels', img_p))
				assert osp.exists(img_path), 'Image not found: %s' % img_path
				assert osp.exists(label_path), 'Label not found: %s' % label_path
				all_img_path.append(img_path)
				all_lbl_path.append(label_path)

	if useVal:
		for pd in val_paths:
			img_dir = osp.join(val_home, pd, 'Images')
			img_paths = sorted(os.listdir(img_dir))
			for img_p in img_paths:
				img_path = osp.abspath(osp.join(img_dir, img_p))
				label_path = osp.abspath(osp.join(val_home, pd, 'Labels', img_p))
				assert osp.exists(img_path), 'Image not found: %s' % img_path
				assert osp.exists(label_path), 'Label not found: %s' % label_path
				all_img_path.append(img_path)
				all_lbl_path.append(label_path)

	assert len(all_img_path) == len(all_lbl_path), (
		'Image count (%d) and label count (%d) do not match.'
		% (len(all_img_path), len(all_lbl_path))
	)
	print('Number of image-label pairs:', len(all_img_path))
	with open('./img_lbl_pair.txt', 'w') as f:
		for img, lbl in zip(all_img_path, all_lbl_path):
			f.write(img + ' ' + lbl + '\n')


def writeTestPredImageLabelPathPairsToTxtFile(
	data_home='./', useTest=True, useVal=False, target_dir=None
):
	"""Write image–prediction path pairs for the test and/or validation splits.

	Pairs each image in the test (or validation) ``Images/`` folder with its
	expected prediction file path under *target_dir*.  The prediction files
	do not need to exist yet; this function is intended to pre-generate the
	list that an inference script will populate.

	Output files:
	    * ``./test_pred_pair.txt``  — when *useTest* is ``True``.
	    * ``./valid_pred_pair.txt`` — when *useVal*  is ``True``.

	Args:
	    data_home (str, optional): Root dataset directory.  Defaults to
	        ``'./'``.
	    useTest (bool, optional): Include the test split.  Defaults to
	        ``True``.
	    useVal (bool, optional): Include the validation split paired with
	        predictions from *target_dir*.  Defaults to ``False``.
	    target_dir (str or None): Root directory where predictions will be
	        stored, following the ``seq<N>/Labels/`` structure.  Required when
	        *useTest* or *useVal* is ``True``.

	Raises:
	    AssertionError: If neither *useTest* nor *useVal* is ``True``, or if
	        any expected source image file does not exist.
	"""
	assert useTest or useVal, 'Error: at least one of useTest or useVal must be True.'

	def _collect_pairs(split_home, pred_root, out_file):
		"""Scan *split_home* and write image/pred pairs to *out_file*."""
		seq_paths = sorted(os.listdir(split_home))
		all_img_path = []
		all_pred_path = []
		for pd in seq_paths:
			img_dir = osp.join(split_home, pd, 'Images')
			img_paths = sorted(os.listdir(img_dir))
			pred_dir = osp.join(pred_root, pd, 'Labels')
			for img_p in img_paths:
				img_path = osp.abspath(osp.join(img_dir, img_p))
				pred_path = osp.abspath(osp.join(pred_dir, img_p))
				assert osp.exists(img_path), 'Image not found: %s' % img_path
				all_img_path.append(img_path)
				all_pred_path.append(pred_path)
		assert len(all_img_path) == len(all_pred_path), (
			'Image count (%d) and prediction path count (%d) do not match.'
			% (len(all_img_path), len(all_pred_path))
		)
		print('Number of image-prediction pairs:', len(all_img_path))
		with open(out_file, 'w') as f:
			for img, pred in zip(all_img_path, all_pred_path):
				f.write(img + ' ' + pred + '\n')

	if useTest:
		_collect_pairs(osp.join(data_home, 'test'), target_dir, './test_pred_pair.txt')
	if useVal:
		_collect_pairs(
			osp.join(data_home, 'valid'), target_dir, './valid_pred_pair.txt'
		)


def parseArgs():
	"""Parse command-line arguments.

	Returns:
	    argparse.Namespace: Parsed arguments with attributes:
	        ``data_home``, ``useTrain``, ``useValid``, ``useTest``, and
	        ``pred_dir``.
	"""
	parser = argparse.ArgumentParser(
		description='Write UAVid image–label path pairs to text files for use '
		'in training or inference pipelines.'
	)
	parser.add_argument(
		'-d',
		dest='data_home',
		type=str,
		default='./',
		help='Dataset root directory (default: ./).',
	)
	parser.add_argument(
		'-t',
		dest='useTrain',
		action='store_true',
		help='Include the training split (train/val mode).',
	)
	parser.add_argument(
		'-v', dest='useValid', action='store_true', help='Include the validation split.'
	)
	parser.add_argument(
		'--test',
		dest='useTest',
		action='store_true',
		help='Write test-image → prediction-path pairs (requires --pred-dir).',
	)
	parser.add_argument(
		'--pred-dir',
		dest='pred_dir',
		type=str,
		default=None,
		help='Prediction root directory used in test/pred mode.',
	)
	return parser.parse_args()


if __name__ == '__main__':
	args = parseArgs()
	# Route to the appropriate function based on the presence of --test / --pred-dir.
	if args.useTest or (args.useValid and args.pred_dir is not None):
		writeTestPredImageLabelPathPairsToTxtFile(
			args.data_home, args.useTest, args.useValid, args.pred_dir
		)
	else:
		writeTrainValImageLabelPathPairsToTxtFile(
			args.data_home, args.useTrain, args.useValid
		)
