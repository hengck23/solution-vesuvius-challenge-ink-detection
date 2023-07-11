from common import *
from share.dataset.dataset import *
from share.dataset.augmentation import *


#############################################################


def train_augment_v2(d, cfg):
	height, width, depth = d.volume.shape
	crop_size = cfg.crop_size

	if np.random.rand() < 0.5:
		volume, label, mask = do_random_crop(
			d.volume,
			d.label,
			d.mask,
			crop_size=crop_size,
			valid_threshold=cfg.valid_threshold,
		)
	else:
		volume, label, mask = do_random_affine_crop(
			d.volume,
			d.label,
			d.mask,
			crop_size=crop_size,
			scale=(0.5, 1.8),
			aspect=(0.5, 1 / 0.5),
			degree=(-45, 45),
			image_fill=0,
			mask_fill=0,
			valid_threshold=cfg.valid_threshold,
		)

	# ----
	if np.random.rand() < 0.5:
		volume = cv2.flip(volume, 1)
		label = cv2.flip(label, 1)
		mask = cv2.flip(mask, 1)

	if np.random.rand() < 0.5:
		volume = cv2.flip(volume, 0)
		label = cv2.flip(label, 0)
		mask = cv2.flip(mask, 0)

	if 1:
		k = np.random.choice(4)
		volume = np.rot90(volume, k, axes=(0, 1))
		label = np.rot90(label, k, axes=(0, 1))
		mask = np.rot90(mask, k, axes=(0, 1))

	if np.random.rand() < 0.5:  # add label noise
		k = np.random.choice(4)
		label = np.rot90(label, k, axes=(0, 1))

	# ---
	if np.random.rand() < 0.5:
		volume, label, mask = do_random_roll(volume, label, mask)
		if (np.random.rand() < 0.2):
			volume, label, mask = do_random_roll(volume, label, mask)

	if np.random.rand() < 0.2:
		volume = do_random_noise(volume, n=[-0.08, 0.08])

	if np.random.rand() < 0.5:
		volume, label = do_random_cutout(
			volume,
			label,
			num_block=3,
			block_size=[40 / cfg.crop_size, 60 / cfg.crop_size],
			mask_fill=0,
		)

	if np.random.rand() < 0.5:
		volume = do_random_contrast(volume)

	# ----
	volume = np.ascontiguousarray(volume)
	label = np.ascontiguousarray(label)
	mask = np.ascontiguousarray(mask)
	return volume, label, mask





#############################################################

class InkDataset(Dataset):
	def __init__(self, data, augment, cfg):

		if 1:
			size = cfg.crop_size
			stride = cfg.crop_size // 2

			length = 0
			for d in data:
				H, W, D = d.volume.shape
				x = np.arange(0, W - size + 1, stride)
				y = np.arange(0, H - size + 1, stride)
				length += len(x) * len(y)

		self.cfg = cfg
		self.length = length
		self.data = data
		self.augment = augment

	def __len__(self):
		return self.length

	def __str__(self):
		num_data = len(self.data)
		string = ''
		string += f'\tlen = {len(self)}\n'
		string += f'\tnum_data = {num_data}\n'
		string += f'\tcrop_id = {[d.crop_id for d in self.data]}\n'
		string += f'\tH,W,D = {[d.volume.shape for d in self.data]}\n'
		string += f'\tink = {[d.label.sum()/d.mask.sum() for d in self.data]}\n'
		return string

	def __getitem__(self, index):
		i = np.random.choice(len(self.data))
		d = self.data[i]

		if self.augment is not None:
			volume, label, mask = self.augment(d, self.cfg)

		r = {}
		r['index'] = index
		r['volume'] = torch.from_numpy(volume / 255).float()
		r['label'] = torch.from_numpy(label).float()
		return r


tensor_key = ['volume', 'label', ]
def null_collate(batch):
	batch_size = len(batch)
	d = {}
	key = batch[0].keys()
	for k in key:
		d[k] = [b[k] for b in batch]

	d['volume'] = torch.stack(d['volume']).permute(0, 3, 1, 2).contiguous()
	d['label'] = torch.stack(d['label']).unsqueeze(1)
	return d


#################################################################################

def run_check_dataset():
	#
	cfg = dotdict(
		valid_threshold=0.8,
		crop_size=128,
		crop_depth=32,
		read_fragment_z = [
			32-16,
			32+16,
		]
	)

	train_data, valid_data = make_fold('2bb', cfg)
	print_data(train_data[0])

	check_train_valid_data(train_data, valid_data, wait=1)
	cv2.waitKey(1)

	# data1 = read_data(1)
	# print_data(data1)
	# print('')
	# train_data = [data1]

	# ------

	dataset = InkDataset(train_data, augment=train_augment_v2, cfg=cfg)
	print(dataset)

	for i in range(5):
		i = np.random.choice(len(dataset))
		r = dataset[i]
		print('index', r['index'], '--------------------')

		for k in tensor_key:
			v = r[k]
			print(k)
			print('\t', 'dtype:', v.dtype)
			print('\t', 'shape:', v.shape)
			if len(v) != 0:
				print('\t', 'min/max:', v.min().item(), '/', v.max().item())
				print('\t', 'is_contiguous:', v.is_contiguous())
				print('\t', 'values:')
				print('\t\t', v.reshape(-1)[:5].data.numpy().tolist(), '...')
				print('\t\t', v.reshape(-1)[-5:].data.numpy().tolist())
		print('')
		if 0:
			# draw
			label = r['label'].data.cpu().numpy()
			volume = r['volume'].data.cpu().numpy()

			index = r['index']
			i, j = dataset.index[index]
			d = dataset.data[i]
			y, x = d.sampling.coord[0][j], d.sampling.coord[1][j]
			size = CFG.crop_size
			x = x * size
			y = y * size
			print(i, x, y)
			d = dataset.data[i]
			ir = d.ir
			cv2.rectangle(ir, (x, y), (x + size, y + size), 0, thickness=12)
			cv2.rectangle(ir, (x, y), (x + size, y + size), 1, thickness=10)
			image_show_norm('ir', ir, min=0, max=1, resize=0.1)

			volume = np.vstack([
				np.hstack(volume[..., k] for k in [0, 1, 2, 3]),
				np.hstack(volume[..., k] for k in [4, 5, 6, 7]),
			])

			image_show_norm('volume', volume, min=0, max=1, resize=4)
			image_show_norm('label', label, min=0, max=1, resize=4)
			cv2.waitKey(0)

	loader = DataLoader(
		dataset,
		sampler=SequentialSampler(dataset),
		# sampler=RandomSampler(dataset),
		# sampler=BalanceSampler(dataset),
		batch_size=5,
		drop_last=True,
		num_workers=0,
		pin_memory=False,
		worker_init_fn=lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
		collate_fn=null_collate,
	)
	print(f'batch_size   : {loader.batch_size}')
	print(f'len(loader)  : {len(loader)}')
	print(f'len(dataset) : {len(dataset)}')
	print('')

	for t, batch in enumerate(loader):
		if t > 5: break
		print('batch ', t, '===================')
		print('index', batch['index'])

		for k in tensor_key:
			v = batch[k]

			print(f'{k}:')
			print('\t', v.data.shape)
			print('\t', 'is_contiguous:', v.is_contiguous())
			if k == 'label':
				print('\t', v.flatten(1).mean(-1))

		if 1:
			pass
		print('')


# main #################################################################
if __name__ == '__main__':
	run_check_dataset()