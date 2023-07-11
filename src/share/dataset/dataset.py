from common import *

import PIL.Image as Image
Image.MAX_IMAGE_PIXELS = 10000000000  # Ignore PIL warnings about large images


train_dir = TRAIN_DIR
crop_region = {
	'1': None,  # (8181, 6330)
	'2': None,  # (14830, 9506)
	'3': None,  # (7606, 5249)

	'1a': [0, 4560, 0, 6330],  # y0,y1,x0,x1
	'1b': [4560, 8181, 0, 6330],
	'2a': [0, 9456, 0, 9506],
	'2b': [9456, 14830, 0, 9506],
	'3a': [0, 4060, 0, 5249],
	'3b': [4060, 7606, 0, 5249],

	'2aa': [0, 7074, 0, 9506],
	'2bb': [7074, 10681, 0, 9506],
	'2cc': [10681, 14830, 0, 9506],

	'3aa': [0, 2216, 0, 5249],
	'3bb': [2216, 3824, 0, 5249],
	'3cc': [3824, 7606, 0, 5249],
}
fragment_data = {
	'1': None,
	'2': None,
	'3': None,
}


# ------------
def do_binarise(m, threshold=0.5):
	m = m - m.min()
	m = m / (m.max() + 1e-7)
	m = (m > threshold).astype(np.float32)
	return m


# ------------
def read_data(fragment_id, z0, z1):
	# volume = np.load(f'{train_dir}/{i}/surface_volume{i}.8.npy')
	volume = []
	start_timer = timer()
	for j in range(z0, z1):
		v = np.array(Image.open(f'{train_dir}/train/{fragment_id}/surface_volume/{j:02d}.tif'), dtype=np.uint16)
		v = (v >> 8).astype(np.uint8)
		# v = (v / 65535.0 * 255).astype(np.uint8)
		volume.append(v)
		print(f'\r @ read volume{j}  {time_to_str(timer() - start_timer, "sec")}', end='', flush=True)
	print('')
	volume = np.stack(volume, -1)
	height, width, depth = volume.shape
	print(f'fragment_id={fragment_id} volume: {volume.shape}')

	# ---
	ir    = cv2.imread(f'{train_dir}/train/{fragment_id}/ir.png', cv2.IMREAD_GRAYSCALE)
	label = cv2.imread(f'{train_dir}/train/{fragment_id}/inklabels.png', cv2.IMREAD_GRAYSCALE)
	mask  = cv2.imread(f'{train_dir}/train/{fragment_id}/mask.png', cv2.IMREAD_GRAYSCALE)


	# ----
	ir = ir / 255
	label = do_binarise(label)
	mask = do_binarise(mask)
	if 0:
		image_show_norm(f'ir', ir, min=0, max=1, resize=0.1)
		image_show_norm(f'label', label, min=0, max=1, resize=0.1)
		cv2.waitKey(0)

	d = dotdict(
		fragment_id=fragment_id,
		crop_id=fragment_id,
		volume=volume,
		ir=ir,
		label=label,
		mask=mask,
	)
	return d


def crop_data(d, region, crop_id):
	y0, y1, x0, x1 = region
	data = dotdict(
		fragment_id=d.fragment_id,
		crop_id=crop_id,
		volume=d.volume[y0:y1, x0:x1],
		ir=d.ir[y0:y1, x0:x1],
		label=d.label[y0:y1, x0:x1],
		mask=d.mask[y0:y1, x0:x1],
	)
	return data

def merge_data(d, crop_id): #vertical merge
	data = dotdict(
		fragment_id = d[0].fragment_id,
		crop_id = crop_id,
		volume  = np.vstack([dd.volume for dd in d]),
		ir      = np.vstack([dd.ir for dd in d]),
		label   = np.vstack([dd.label for dd in d]),
		mask    = np.vstack([dd.mask for dd in d]),
	)
	return data



def read_crop_data(crop_id, z0, z1):
	fragment_id = crop_id[:1]
	if fragment_data[fragment_id] is None:
		fragment_data[fragment_id] = read_data(fragment_id, z0, z1)

	if fragment_id == crop_id:
		d = fragment_data[fragment_id]
	else:
		d = crop_data(fragment_data[fragment_id], crop_region[crop_id], crop_id)
	return d


def print_data(d):
	print('fragment_id:', d.fragment_id)
	print('crop_id:', d.crop_id)
	print('volume :', d.volume.shape, d.volume.min(), d.volume.max())
	print('ir     :', d.ir.shape, d.ir.min(), d.ir.max())
	print('label  :', d.label.shape, d.label.min(), d.label.max())
	print('mask   :', d.mask.shape, d.mask.min(), d.mask.max())


# d = read_crop_data('1')
# d = read_crop_data('1a')
# print_data(d)
# exit(0)


def make_fold(fold, CFG):

	split = {
		'1': dotdict(
			valid_id=['1'],
			train_id=['2', '3'],
		),
		'2': dotdict(
			valid_id=['2'],
			train_id=['1', '3'],
		),
		'3': dotdict(
			valid_id=['3'],
			train_id=['1', '2'],
		),

		# ---
		'1a': dotdict(
			valid_id=['1a'],
			train_id=['1b', '2', '3'],
		),
		'1b': dotdict(
			valid_id=['1b'],
			train_id=['1a', '2', '3'],
		),
		'2a': dotdict(
			valid_id=['2a'],
			train_id=['1', '2b', '3'],
		),
		'2b': dotdict(
			valid_id=['2b'],
			train_id=['1', '2a', '3'],
		),
		'3a': dotdict(
			valid_id=['3a'],
			train_id=['1', '2', '3b'],
		),
		'3b': dotdict(
			valid_id=['3b'],
			train_id=['1', '2', '3a'],
		),

		# ---
		'2aa': dotdict(
			valid_id=['2aa'],
			train_id=['1', ['2bb', '2cc'], '3'],
		),
		'2bb': dotdict(
			valid_id=['2bb'],
			train_id=['1', ['2aa', '2cc'], '3'],
		),
		'2cc': dotdict(
			valid_id=['2cc'],
			train_id=['1', ['2aa', '2bb'], '3'],
		),

		# ---
		'3aa': dotdict(
			valid_id=['3aa'],
			train_id=['1', '2', ['3bb','3cc']],
		),
		'3bb': dotdict(
			valid_id=['3bb'],
			train_id=['1', '2', ['3aa', '3cc']],
		),
		'3cc': dotdict(
			valid_id=['3cc'],
			train_id=['1', '2', ['3aa','3bb']],
		),
	}
	z0 = CFG.read_fragment_z[0]
	z1 = CFG.read_fragment_z[1]
	valid_data = [read_crop_data(i, z0, z1) for i in split[fold].valid_id]
	train_data = []
	for i in split[fold].train_id:
		if not isinstance(i, list):
			train_data.append(read_crop_data(i, z0, z1))
		else:
			train_data.append(merge_data(
				[read_crop_data(j, z0, z1) for j in i], '+'.join(i)
			))

	return train_data, valid_data


# make sure they don't overlap
def check_train_valid_data(train_data, valid_data, wait=0):
	for d in train_data:
		image_show_norm(f'tain-{d.crop_id}', d.ir, min=0, max=1, resize=0.025)
	# cv2.waitKey(1)

	for d in valid_data:
		image_show_norm(f'valid-{d.crop_id}', d.ir, min=0, max=1, resize=0.025)
	cv2.waitKey(wait)



# train_data, valid_data = make_fold(fold='2bb')
# check_train_valid_data(train_data, valid_data, wait=0)
# exit(0)

################################################################################################################



