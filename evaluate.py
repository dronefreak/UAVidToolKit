"""Evaluation script for UAVid semantic segmentation predictions.

Computes pixel-wise Intersection over Union (IoU) and overall pixel accuracy
by accumulating a confusion matrix across all ground-truth / prediction image
pairs in the dataset directory tree.

Expected directory layout
-------------------------
::

    <gt_dir>/
        seq<N>/
            Labels/
                <frame>.png  ← 3-channel RGB or 1-channel train-ID label

    <pred_dir>/
        seq<N>/
            Labels/
                <frame>.png  ← same structure, same filenames

Usage
-----
::

    python evaluate.py -gt <ground_truth_dir> -p <prediction_dir> [-v] [-o <output_dir>]

Options
-------
-gt     Ground-truth directory.
-p      Prediction directory.
-v      Produce and save bar-chart (IoU per class) and confusion-matrix PNG.
-o      Output directory for saved figures (default: current directory).
"""

import os
import os.path as osp
import numpy as np
import argparse
import itertools
from tqdm import tqdm
from colorTransformer import UAVidColorTransformer
from PIL import Image
from matplotlib import pyplot as plt

# ---------------------------------------------------------------------------
# C/Cython acceleration
# ---------------------------------------------------------------------------
# The Cython extension is preferred for large datasets; scikit-learn is the
# pure-Python fallback when the extension has not been compiled.
CSUPPORT = True
if CSUPPORT:
	try:
		import addToConfusionMatrix
	except ImportError:
		CSUPPORT = False
		from sklearn import metrics

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
#: Set of image file extensions that are considered valid label files.
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

# Build the global colour transformer and derive ordered label/name tuples
# from the colour table so that LABELS[i] == i and CLASS_NAMES[i] is the
# human-readable name for class i.
clr_trans = UAVidColorTransformer()
LABELS, CLASS_NAMES = zip(
	*[[i, name] for i, name in enumerate(clr_trans.colorTable().keys())]
)


# ---------------------------------------------------------------------------
# Confusion-matrix construction
# ---------------------------------------------------------------------------


def getConfusionMatrixForImageList(
	classNum, predfileList, truefileList, evalLabels=LABELS
):
	"""Accumulate a confusion matrix over a list of prediction/GT image pairs.

	Both RGB colour images (3-channel) and single-channel train-ID images are
	accepted.  RGB images are automatically converted to train-IDs before
	accumulation.

	Args:
	    classNum (int): Number of classes (sets the matrix dimension).
	    predfileList (list[str]): Ordered list of prediction image file paths.
	    truefileList (list[str]): Ordered list of ground-truth image file paths.
	        Must be the same length as *predfileList*.
	    evalLabels (tuple[int], optional): Class IDs to include in the matrix.
	        Defaults to :data:`LABELS`.

	Returns:
	    numpy.ndarray: ``classNum × classNum`` uint64 confusion matrix where
	    ``cm[gt, pred]`` holds the pixel count for that (GT class, predicted
	    class) pair.
	"""
	assert len(predfileList) == len(truefileList), (
		'predfileList and truefileList must have the same length.'
	)
	enc = clr_trans
	cm = np.zeros(shape=[classNum, classNum], dtype=np.uint64)
	print('CSUPPORT:', CSUPPORT)
	if CSUPPORT:
		print('Using fast Cython/C++ evaluation.')
	else:
		print(
			'Using slow pure-Python evaluation (compile the Cython extension for speed).'
		)
	for idx in tqdm(range(len(predfileList))):
		predfile = predfileList[idx]
		truefile = truefileList[idx]
		imagePred = np.array(Image.open(predfile))
		imageTrue = np.array(Image.open(truefile))
		# Convert RGB colour images to single-channel train-ID images.
		if len(imagePred.shape) == 3 and len(imageTrue.shape) == 3:
			imagePred = enc.transform(imagePred, dtype=np.uint8)
			imageTrue = enc.transform(imageTrue, dtype=np.uint8)
		assert len(imagePred.shape) == 2 and len(imageTrue.shape) == 2, (
			'After optional RGB→ID conversion both images must be 2-D.'
		)
		if CSUPPORT:
			cm = addToConfusionMatrix.cEvaluatePair(
				imagePred, imageTrue, cm, evalLabels
			)
		else:
			cm = calculateConfusionMatrix(cm, imagePred, imageTrue, evalLabels)
	return cm


def calculateConfusionMatrix(cm, imagePred, imageTrue, evalLabels):
	"""Update *cm* with the sklearn confusion matrix for one image pair.

	This is the pure-Python fallback used when the Cython extension is not
	available.  It is significantly slower than the C++ path for large images.

	Args:
	    cm (numpy.ndarray or None): Existing confusion matrix to add to, or
	        ``None`` to start fresh.
	    imagePred (numpy.ndarray): 2-D array of predicted train-IDs.
	    imageTrue (numpy.ndarray): 2-D array of ground-truth train-IDs.
	    evalLabels (tuple[int]): Class IDs used as the confusion-matrix axes.

	Returns:
	    numpy.ndarray: Updated confusion matrix (same shape as the sklearn
	    output for *evalLabels*).
	"""
	y_true = np.reshape(imageTrue, [-1, 1])
	y_pred = np.reshape(imagePred, [-1, 1])
	cm_t = metrics.confusion_matrix(y_true, y_pred, labels=evalLabels)
	if cm is None:
		return cm_t
	return cm + cm_t


# ---------------------------------------------------------------------------
# IoU metrics
# ---------------------------------------------------------------------------


def getIouScoreForLabel(label, cm):
	"""Compute the Intersection-over-Union (IoU) score for a single class.

	IoU = TP / (TP + FP + FN).  Returns ``float('nan')`` when the class is
	absent from both the ground truth and the predictions (denominator == 0),
	so that it can be safely excluded from the mean with :func:`numpy.nanmean`.

	Args:
	    label (int): Class train-ID (row/column index into *cm*).
	    cm (numpy.ndarray): Square confusion matrix, where ``cm[gt, pred]``
	        is the pixel count for that pair.

	Returns:
	    float: IoU score in ``[0, 1]``, or ``float('nan')`` if the class is
	    absent.
	"""
	tp = np.longlong(cm[label, label])
	fn = np.longlong(cm[label, :].sum()) - tp  # GT positives predicted as other classes
	fp = (
		np.longlong(cm[:, label].sum()) - tp
	)  # Other GT classes predicted as this class
	denom = tp + fp + fn
	if denom == 0:
		return float('nan')
	return float(tp) / denom


def getMeanIOU(cm, evallabels=LABELS):
	"""Compute the mean IoU across all evaluation classes, ignoring absent ones.

	Classes for which the denominator (TP+FP+FN) is zero are excluded from
	the mean via :func:`numpy.nanmean`.

	Args:
	    cm (numpy.ndarray): Square confusion matrix.
	    evallabels (tuple[int], optional): Class IDs to evaluate.
	        Defaults to :data:`LABELS`.

	Returns:
	    float: Mean IoU score in ``[0, 1]``.
	"""
	IOUs = [getIouScoreForLabel(l, cm) for l in evallabels]
	return np.nanmean(IOUs)


def getIOUforClasses(cm, evallabels=LABELS):
	"""Return a per-class list of IoU scores in evaluation-label order.

	Args:
	    cm (numpy.ndarray): Square confusion matrix.
	    evallabels (tuple[int], optional): Class IDs to evaluate.
	        Defaults to :data:`LABELS`.

	Returns:
	    list[float]: IoU score for each class in *evallabels* order.
	        Absent classes are represented as ``float('nan')``.
	"""
	return [getIouScoreForLabel(l, cm) for l in evallabels]


def getClassPixelCounts(cm):
	"""Return the total ground-truth pixel count for each class.

	Derived from the row sums of the confusion matrix (each row corresponds
	to one GT class).  Useful for diagnosing class imbalance.

	Args:
	    cm (numpy.ndarray): Square confusion matrix with :data:`CLASS_NAMES`
	        ordering.

	Returns:
	    dict[str, int]: Mapping of ``{class_name: gt_pixel_count}``.
	"""
	return {CLASS_NAMES[i]: int(cm[i, :].sum()) for i in range(len(CLASS_NAMES))}


# ---------------------------------------------------------------------------
# Directory-level helpers
# ---------------------------------------------------------------------------


def getConfusionMatrixfromDirectory(gt_home, pred_home):
	"""Build a confusion matrix by scanning two mirrored directory trees.

	Scans *gt_home* for sequence subdirectories (prefixed with ``'seq'``),
	collects all label images, and matches them to corresponding prediction
	images in *pred_home*.  Raises :class:`AssertionError` if any expected
	prediction file is missing.

	Args:
	    gt_home (str): Root directory containing GT sequence folders.
	    pred_home (str): Root directory containing prediction sequence folders
	        with the same structure as *gt_home*.

	Returns:
	    numpy.ndarray: Accumulated confusion matrix across all image pairs.
	"""
	gt_paths = sorted([p for p in os.listdir(gt_home) if p.startswith('seq')])
	all_gt_path = []
	all_pred_path = []
	for pd in gt_paths:
		lbl_dir = osp.join(gt_home, pd, 'Labels')
		lbl_paths = sorted(
			[
				p
				for p in os.listdir(lbl_dir)
				if osp.splitext(p)[1].lower() in IMAGE_EXTENSIONS
			]
		)
		for lbl_p in lbl_paths:
			lbl_path = osp.abspath(osp.join(lbl_dir, lbl_p))
			pred_path = osp.join(pred_home, pd, 'Labels', lbl_p)
			assert osp.exists(lbl_path)
			assert osp.exists(pred_path), (
				'Prediction incomplete, cannot find prediction: %s' % pred_path
			)
			all_gt_path.append(lbl_path)
			all_pred_path.append(pred_path)
	class_num = len(CLASS_NAMES)
	return getConfusionMatrixForImageList(class_num, all_pred_path, all_gt_path)


# ---------------------------------------------------------------------------
# Pixel accuracy and matrix normalisation
# ---------------------------------------------------------------------------


def getPixelAccuracy(cm):
	"""Compute overall pixel accuracy from a confusion matrix.

	Pixel accuracy = (sum of diagonal) / (sum of all elements).

	Args:
	    cm (numpy.ndarray): Square confusion matrix.

	Returns:
	    float: Pixel accuracy in ``[0, 1]``.
	"""
	return np.trace(cm).astype(np.float64) / cm.sum()


def normalize_confusion_matrix(cm):
	"""Row-normalise a confusion matrix so each row sums to 1.

	Rows with a zero sum (classes absent from the GT) are left as all-zeros
	rather than producing NaN or Inf.

	Args:
	    cm (numpy.ndarray): Square confusion matrix (integer or float).

	Returns:
	    numpy.ndarray: float32 row-normalised confusion matrix of the same
	    shape.
	"""
	cm = cm.astype(np.float32)
	row_sums = cm.sum(axis=1, keepdims=True)
	# Guard against division by zero for classes with no GT pixels.
	row_sums[row_sums == 0] = 1
	return cm / row_sums


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def visualizeMeanIOUforClasses(
	IOUs, labels, classNames, prefix='', suffix='', output_dir='./'
):
	"""Save a bar chart of per-class IoU scores with a final MeanIoU bar.

	Each bar is coloured with the class's dataset colour.  Numeric IoU values
	are annotated above each bar.  The chart is saved as a PNG to *output_dir*
	rather than displayed interactively.

	Args:
	    IOUs (list[float]): Per-class IoU scores, one per entry in *labels*.
	    labels (tuple[int]): Class train-IDs in the same order as *IOUs*.
	    classNames (tuple[str]): Class names in the same order as *labels*.
	    prefix (str, optional): Filename prefix for the saved PNG.
	    suffix (str, optional): Filename suffix inserted before the extension.
	    output_dir (str, optional): Directory where the PNG is saved.
	        Defaults to the current working directory.
	"""
	assert len(IOUs) == len(labels)
	assert len(IOUs) == len(classNames)
	y_pos = np.arange(len(IOUs) + 1)
	plt.figure(figsize=(14, 5))
	mean_iou = np.nanmean(IOUs)
	barlist = plt.bar(y_pos, IOUs + [mean_iou], align='center')
	ax = plt.axes()
	color_tab = clr_trans.colorTable()
	for idx, l in enumerate(labels):
		# Normalise the dataset RGB colour to matplotlib's [0, 1] range.
		clr = color_tab[classNames[l]]
		clr_face = [clr[0] / 255.0, clr[1] / 255.0, clr[2] / 255.0, 1.0]
		barlist[idx].set_color(clr_face)
		barlist[idx].set_edgecolor(clr_face)
		height = barlist[idx].get_height()
		ax.text(
			barlist[idx].get_x() + barlist[idx].get_width() / 2.0,
			height + 0.02,
			'%.3f' % (IOUs[idx]),
			ha='center',
			va='bottom',
		)
	# Annotate the final MeanIoU bar.
	height = barlist[-1].get_height()
	ax.text(
		barlist[-1].get_x() + barlist[-1].get_width() / 2.0,
		height + 0.02,
		'%.3f' % mean_iou,
		ha='center',
		va='bottom',
	)
	ax.yaxis.grid(color=(0.6, 0.6, 0.6), linestyle='--')
	plt.xticks(y_pos, [n.replace('_', ' ') for n in classNames + ('MeanIOU',)])
	plt.tick_params(axis='x', which='major', labelsize=14)
	plt.ylabel('IOU Scores', fontsize=13)
	plt.ylim(0, 1)
	plt.savefig(os.path.join(output_dir, prefix + 'IOU_scores' + suffix + '.png'))


def visualizeConfusionMatrix(
	cm,
	class_names,
	normalize=False,
	title='Confusion matrix',
	cmap=plt.cm.Blues,
	prefix='',
	suffix='',
	output_dir='./',
):
	"""Save a colour-coded confusion matrix as a PNG image.

	Cell text is white for high-value cells and black for low-value cells to
	ensure readability.  The figure is saved to *output_dir*.

	Args:
	    cm (numpy.ndarray): Square confusion matrix.
	    class_names (tuple[str]): Class name for each row/column.
	    normalize (bool, optional): If ``True``, normalise each row to sum to
	        1.0 before plotting.  Defaults to ``False``.
	    title (str, optional): Figure title and output filename stem.
	        Defaults to ``'Confusion matrix'``.
	    cmap (matplotlib.colors.Colormap, optional): Colour map applied to the
	        matrix cells.  Defaults to ``plt.cm.Blues``.
	    prefix (str, optional): Filename prefix for the saved PNG.
	    suffix (str, optional): Filename suffix inserted before the extension.
	    output_dir (str, optional): Directory where the PNG is saved.
	        Defaults to the current working directory.
	"""
	if normalize:
		row_sums = cm.sum(axis=1, keepdims=True).astype('float')
		row_sums[row_sums == 0] = 1  # guard against zero-GT classes
		cm = cm.astype('float') / row_sums
		print('Normalized confusion matrix')
	else:
		print('Confusion matrix, without normalization')

	plt.figure(figsize=(8, 6))
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(class_names))
	class_names = [n.replace('_', ' ') for n in class_names]
	plt.xticks(tick_marks, class_names, rotation=45)
	plt.yticks(tick_marks, class_names)
	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.0
	# Annotate every cell with its numeric value; use white text on dark cells.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(
			j,
			i,
			format(cm[i, j], fmt),
			horizontalalignment='center',
			fontsize=6,
			color='white' if cm[i, j] > thresh else 'black',
		)
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, prefix + title + suffix + '.png'))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parseArgs():
	"""Parse command-line arguments for the evaluation script.

	Returns:
	    argparse.Namespace: Parsed arguments with attributes:
	        ``gt_dir``, ``pred_dir``, ``use_visualize``, ``output_dir``.
	"""
	parser = argparse.ArgumentParser(
		description='Evaluate UAVid segmentation predictions.'
	)
	parser.add_argument(
		'-gt', dest='gt_dir', help='Ground-truth directory (contains seq<N>/Labels/).'
	)
	parser.add_argument(
		'-p', dest='pred_dir', help='Prediction directory (same structure as gt_dir).'
	)
	parser.add_argument(
		'-v',
		dest='use_visualize',
		action='store_true',
		help='Save IoU bar-chart and confusion matrix as PNG files.',
	)
	parser.add_argument(
		'-o',
		dest='output_dir',
		default='./',
		help='Output directory for saved figures (default: ./).',
	)
	return parser.parse_args()


def evaluateFromDirectories(args):
	"""Run the full evaluation pipeline and print results to stdout.

	Builds a confusion matrix from all matched image pairs, computes per-class
	IoU, mean IoU, pixel accuracy, and per-class GT pixel counts.  Optionally
	saves visualisation figures.

	Args:
	    args (argparse.Namespace): Parsed CLI arguments as returned by
	        :func:`parseArgs`.
	"""
	gt_home = args.gt_dir
	pred_home = args.pred_dir
	use_visualize = args.use_visualize
	output_dir = args.output_dir

	cm = getConfusionMatrixfromDirectory(gt_home, pred_home)
	IOUs = getIOUforClasses(cm, evallabels=LABELS)
	mIOU = getMeanIOU(cm)
	acc = getPixelAccuracy(cm)
	pixel_counts = getClassPixelCounts(cm)

	iou_dict = {name: iou for name, iou in zip(CLASS_NAMES, IOUs)}
	print('IOUs:', iou_dict)
	print('mIOU:', mIOU)
	print('acc:', acc)
	print('GT pixel counts per class:', pixel_counts)

	if use_visualize:
		visualizeMeanIOUforClasses(
			IOUs, labels=LABELS, classNames=CLASS_NAMES, output_dir=output_dir
		)
		visualizeConfusionMatrix(
			cm,
			CLASS_NAMES,
			title='Confusion Matrix',
			normalize=True,
			output_dir=output_dir,
		)
		plt.show()


if __name__ == '__main__':
	args = parseArgs()
	evaluateFromDirectories(args)
