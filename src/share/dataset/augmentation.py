import cv2
import numpy as np

from common import *
import PIL.Image as Image
'''
 A.RandomBrightnessContrast(p=0.75),
		A.ShiftScaleRotate(p=0.75),
		A.OneOf([
				A.GaussNoise(var_limit=[10, 50]),
				A.GaussianBlur(),
				A.MotionBlur(),
				], p=0.4),
		A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
		A.CoarseDropout(max_holes=1, max_width=int(size * 0.3), max_height=int(size * 0.3), 
						mask_fill_value=0, p=0.5),

'''
## geometric --------------------------------
def do_random_crop(
	image,
	mask,
	valid,
	crop_size=224,
	valid_threshold=0.5,
):
	height, width = image.shape[:2]
	while (1):
		y = np.random.randint(0, height - crop_size+1)
		x = np.random.randint(0, width  - crop_size+1)
		crop_image = image[y:y + crop_size, x:x + crop_size]
		crop_mask  = mask[y:y + crop_size, x:x + crop_size]
		crop_valid = valid[y:y + crop_size, x:x + crop_size]
		if crop_valid.mean() > valid_threshold:
			break
	return crop_image, crop_mask, crop_valid


def do_random_affine_on_point(
	point,
	scale  = (0.5,2.0),
	aspect = (0.9,1/0.9),
	degree = (-45,45),
):
	center = point.mean(0,keepdims=True)
	pt = point - center

	#scale
	s = random.uniform(*scale)
	a = random.uniform(*aspect)
	pt = pt*[[s, s*a]]

	#rotate
	angle = random.uniform(*degree)
	angle = np.radians(angle)
	c, s = np.cos(angle), np.sin(angle)
	rotate = np.array(((c, -s), (s, c)))
	pt = (rotate@pt.T).T
	point = pt + center
	return point

def do_random_affine_crop(
	image,
	mask,
	valid,
	crop_size=224,
	scale  = (0.5,2.0),
	aspect = (0.9,1/0.9),
	degree = (-45,45),
	image_fill=0,
	mask_fill=0,
	valid_threshold=0.5,
):
	height, width = image.shape[:2]
	s = crop_size
	point = np.array([
		[0,0],
		[0,s],
		[s,s],
		[s,0],
	])
	point1 = do_random_affine_on_point(
		point,
		scale  = scale,
		aspect = aspect,
		degree = degree,
	)
	point1 = point1.astype(int)
	x0, y0 = point1[:,0].min(), point1[:,1].min()
	point1 = point1 - [[x0, y0]]
	w, h = point1[:,0].max(), point1[:,1].max()

	while (1):
		y = np.random.randint(0, height - w+1)
		x = np.random.randint(0, width  - h+1)
		crop_valid  = valid[y:y + h, x:x + w]
		if crop_valid.mean() > valid_threshold:
			break

	point2 = point1 + [[x, y]]
	#----
	matrix = cv2.getAffineTransform(
		point2[:3].astype(np.float32),
		point[:3].astype(np.float32),
	)
	crop_image = cv2.warpAffine(
		image, matrix, (crop_size,crop_size),
		flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT, borderValue=image_fill,
	)
	crop_mask = cv2.warpAffine(
		mask, matrix, (crop_size,crop_size),
		flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT, borderValue=mask_fill,
	)
	crop_valid = cv2.warpAffine(
		valid, matrix, (crop_size,crop_size),
		flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT, borderValue=0,
	)
	if 0:
		for i in range(4):
			j=(i+1)%4
			xx0,yy0=point2[i]
			xx1,yy1=point2[j]
			cv2.line(mask,(xx0,yy0),(xx1,yy1), 128, 8)
			cv2.line(image,(xx0,yy0),(xx1,yy1), (255,255,255), 8)

		image_show_norm('image', image, resize=0.05)
		image_show_norm('mask', mask, resize=0.05)
		image_show_norm('crop_image', crop_image, resize=1)
		image_show_norm('crop_mask', crop_mask, resize=1)
		cv2.waitKey(0)

	return crop_image, crop_mask, crop_valid



def do_random_roll(
	image,
	mask,
	valid,
):
	height, width = image.shape[:2]
	y = np.random.uniform(0.2,0.8)
	x = np.random.uniform(0.2,0.8)
	y = int(y*height)
	x = int(x*width)

	image = np.roll(image,y,axis=0)
	mask  = np.roll(mask,y,axis=0)
	valid  = np.roll(valid,y,axis=0)

	image = np.roll(image,x,axis=1)
	mask  = np.roll(mask,x,axis=1)
	valid  = np.roll(valid,x,axis=1)

	return image, mask, valid



## noise --------------------------------


def do_random_cutout(
	image,
	mask,
	num_block=5,
	block_size=[0.1,0.3],
	fill_mode ='constant',
	image_fill=0,
	mask_fill=0,
):
	height, width = image.shape[:2]

	if num_block>1:
		num_block = np.random.randint(num_block-1)+1

	for n in range(num_block):
		s = np.random.uniform(*block_size)
		s = int(s*(height+width)/2)

		x = np.random.randint(0,width-s)
		y = np.random.randint(0,height-s)
		if fill_mode=='constant':
			image[y:y+s,x:x+s]=image_fill
			if mask_fill is not None:
				mask[y:y+s,x:x+s]=mask_fill
		else:
			raise NotImplementedError
	return image, mask


def do_random_noise(
	image,
	n = [-0.2,0.2],
):
	height, width, d = image.shape
	image = image.astype(np.float32)/255
	noise = np.random.uniform(*n,(height, width, d))
	image = image + noise
	image = np.clip(image,0,1)
	image = (image*255).astype(np.uint8)
	return image


def do_random_blur(
	image,
	k = [3,7],
):
	height, width, d = image.shape
	k = np.random.randint(*k)
	k = 2*(k//2)+1
	image = cv2.GaussianBlur(image,(k,k),0)
	return image

## intensity --------------------------------
def do_random_contrast(
	image,
	a = [-0.3,0.3],
	b = [-0.5,0.5],
	c = [-0.2,0.2],
):
	image = image.astype(np.float32)/255
	u = np.random.choice(3)
	if u==0:
		m = np.random.uniform(*a)
		image = image*(1+m)
	if u==1:
		m = np.random.uniform(*b)
		image = image**(1+m)
	if u==2:
		m = np.random.uniform(*c)
		image = image + m

	image = np.clip(image,0,1)
	image = (image*255).astype(np.uint8)
	return image


## debug ###############################################################
def read_debug_data(i):
	# volume = np.load(f'{train_dir}/{i}/surface_volume{i}.8.npy')
	train_dir = f'{root_dir}/data/vesuvius-challenge-ink-detection/train'
	volume = []
	start_timer = timer()
	for j in range(32-2, 32+1):
		v = np.array(Image.open(f'{train_dir}/{i}/surface_volume/{j:02d}.tif'), dtype=np.uint16)
		v = (v >> 8).astype(np.uint8)
		# v = (v / 65535.0 * 255).astype(np.uint8)
		volume.append(v)
		print(f'\r @ read volume{j}  {time_to_str(timer() - start_timer, "sec")}', end='', flush=True)
	print('')
	volume = np.stack(volume, -1)
	height, width, depth = volume.shape
	print(f'fragment_id={i} volume: {volume.shape}')

	# ---
	ir =   cv2.imread(f'{train_dir}/{i}/ir.png', cv2.IMREAD_GRAYSCALE)
	label = cv2.imread(f'{train_dir}/{i}/inklabels.png', cv2.IMREAD_GRAYSCALE)
	mask = cv2.imread(f'{train_dir}/{i}/mask.png', cv2.IMREAD_GRAYSCALE)

	d = dotdict(
		fragment_id=i,
		volume=volume,
		ir=ir,
		label=label,
		mask=mask,
	)
	return d

def make_checker_image(checker_shape,size=5):
	H,W = checker_shape
	small = np.zeros((H,W), np.float32)
	for y in range(H):
		small[y,y%2::2]=1
	checker = cv2.resize(small,dsize=None,fx=size,fy=size,interpolation=cv2.INTER_NEAREST)
	return checker

def draw_affine_matrix(image, crop_shape, matrix):
	ch,cw = crop_shape
	inv_matrix = cv2.invertAffineTransform(matrix)
	dst = np.array([
		0,0,1,
		cw,0,1,
		cw,ch,1,
		0,ch,1,
	]).astype(np.float32).reshape(4,3)
	src = (inv_matrix@dst.T).T
	pt = src
	for i in range(4):
		x1, y1 = pt[i].astype(int)
		x2, y2 = pt[(i + 1) % 4].astype(int)
		cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)
		if i == 0:
			cv2.circle(image, (x1, y1), 10, (0, 255, 0), -1)
		if i == 3:
			cv2.circle(image, (x1, y1), 10, (255, 0, 0), -1)
	return image


def run_check_crop():
	d = read_debug_data(1)

	while 1:
		transform = []
		for t in range(5):
			#volume, label = do_random_crop(d.volume, d.label, crop_size=224)
			volume, label = do_random_affine_crop(
				d.volume,
				d.label,
				crop_size=224,
				scale=(0.8, 1.2),
				aspect=(0.8, 1 / 0.8),
				degree=(-15, 15),
				image_fill=0,
				mask_fill=0,
			)


			label =cv2.cvtColor(label,cv2.COLOR_GRAY2BGR)
			transform.append(np.vstack([volume, label]))
		transform = np.hstack(transform)

		image_show_norm('transform',transform,resize=1)
		cv2.waitKey(0)

def run_check_augment():
	d = read_debug_data(1)
	volume, label = do_random_crop(d.volume, d.label, d.mask, crop_size=224)

	transform = []
	for t in range(5):
		# volume1, label1 = do_random_cutout(
		# 	volume.copy(),
		# 	label.copy(),
		# 	num_block=1,
		# 	block_size=[0.25, 0.35],
		# 	fill_mode='constant',
		# 	fill_value=0,
		# )

		#volume1, label1 = do_random_contrast(volume.copy()),  label
		volume1, label1 = do_random_noise(volume.copy()),  label
		#volume1, label1 = do_random_roll(volume.copy(),  label.copy())

		label1 =cv2.cvtColor(label1,cv2.COLOR_GRAY2BGR)
		cv2.rectangle(volume1,(0,0),(223,223),(255,255,255))
		cv2.rectangle(label1,(0,0),(223,223),(255,255,255))
		transform.append(np.vstack([volume1, label1]))
	transform = np.hstack(transform)

	image_show_norm('transform',transform,resize=1)
	cv2.waitKey(0)

# main #################################################################
if __name__ == '__main__':

	#run_check_crop()
	run_check_augment()

