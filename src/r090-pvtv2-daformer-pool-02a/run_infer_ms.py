import os

import cv2
import numpy as np
import torch

os.environ['CUDA_VISIBLE_DEVICES']='0,1'

from common import *
from my_lib.net.lookahead import *
from my_lib.net.rate import *

from dataset import *
from model import *
from kaggle_ink_v1 import *


#------------------------------------------

#from config_fold1_stage2_0 import *
from config_fold2aa_stage2_0 import *


#------------------------------------------

CFG.stride = CFG.crop_size//2
CFG.tta_rotate = False
CFG.tta_scale = [1, 1.20, 0.80]
print(cfg_to_text())

batch_size = 16
checkpoint = \
	fold_dir + '/checkpoint/fold2aa-Pvt2b3MeanPoolDaformer-00009159.model.pth'  #
	#fold_dir + '/checkpoint/fold1-Pvt2b3MeanPoolDaformer-00029376.model.pth'  #

checkpoint_no = checkpoint.split('.')[-3][-8:]
infer_dir = \
	f'{fold_dir}/infer/{checkpoint_no}-s{CFG.stride}-scale{CFG.tta_scale}-rot{int(CFG.tta_rotate)}'


#################################################################################################
## setup
os.makedirs(infer_dir, exist_ok=True)
log = Logger()
log.open(fold_dir + '/log.infer.txt', mode='a')

def metric_to_text(ink, label, mask):
	text = []

	p = ink.reshape(-1)
	t = label.reshape(-1)
	pos = np.log(np.clip(p,1e-7,1))
	neg = np.log(np.clip(1-p,1e-7,1))
	bce = -(t*pos +(1-t)*neg).mean()
	text.append(f'bce={bce:0.5f}')

	mask_sum = mask.sum()
	#print(f'{threshold:0.1f}, {precision:0.3f}, {recall:0.3f}, {fpr:0.3f},  {dice:0.3f},  {score:0.3f}')
	text.append('p_sum  th   prec   recall   fpr   dice   score')
	text.append('----------------------------------------------')
	for threshold in np.arange(0.05,0.99,0.05):#[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
		p = ink.reshape(-1)
		t = label.reshape(-1)
		p = (p > threshold).astype(np.float32)
		t = (t > 0.5).astype(np.float32)

		tp = p * t
		precision = tp.sum() / (p.sum() + 0.0001)
		recall = tp.sum() / t.sum()

		fp = p * (1 - t)
		fpr = fp.sum() / (1 - t).sum()

		beta = 0.5
		#  0.2*1/recall + 0.8*1/prec
		score = beta * beta / (1 + beta * beta)  / (recall+0.001) + 1 / (1 + beta * beta)  / (precision+0.001)
		score = 1 / score

		dice = 2 * tp.sum() / (p.sum() + t.sum())
		p_sum = p.sum()/mask_sum

		# print(fold, threshold, precision, recall, fpr,  score)
		text.append( f'{p_sum:0.3f}, {threshold:0.2f}, {precision:0.3f}, {recall:0.3f}, {fpr:0.3f},  {dice:0.3f},  {score:0.3f}')
	text = '\n'.join(text)
	return text


# multi scale
def infer_one_ms(net, d):
	infer_mask = make_mask(CFG)

	net = net.cuda()
	net = net.eval()
	net.output_type = ['inference']

	#get coord
	crop_size = CFG.crop_size
	stride = CFG.stride
	H,W,D = d.volume.shape

	probability = np.zeros((H, W))
	count = np.zeros((H, W))

	for s in CFG.tta_scale:#
		print(f'\nscale @ {s}')
		if s==1:
			scale_volume = d.volume
		else:
			scale_volume = cv2.resize(d.volume, dsize=None, fx =s, fy=s)

		sH, sW, D = scale_volume.shape

		##pad #assume H,W >size
		px,py = sW%stride,sH%stride
		if (px!=0) or (py!=0):
			px = stride-px
			py = stride-py
			pad_volume = np.pad(scale_volume, [(0, py), (0, px), (0, 0)], constant_values=0)
		else:
			pad_volume = scale_volume

		pH, pW, _  = pad_volume.shape
		x = np.arange(0,pW-crop_size+1,stride)
		y = np.arange(0,pH-crop_size+1,stride)
		x,y = np.meshgrid(x,y)
		xy  = np.stack([x,y],-1).reshape(-1,2)
		print('H,W -> sH,sW ->  pH,pW,len(xy)',H,W,'->',sH,sW,'->',pH,pW,len(xy))

		batch_iter = []
		for t in range(0,len(xy),batch_size):
			batch_iter.append(xy[t:t+batch_size])

		#---
		scale_probability = np.zeros((pH,pW))
		scale_count = np.zeros((pH,pW))


		start_timer = timer()
		for t, xy0 in enumerate(batch_iter):
			print('\r', t, len(batch_iter), time_to_str(timer() - start_timer,'min'), end='')

			volume =[]
			for x0,y0 in xy0 :
				v = pad_volume[y0:y0 + crop_size, x0:x0 + crop_size]
				volume.append(v)
			volume = np.stack(volume)
			volume = np.ascontiguousarray(volume.transpose(0,3,1,2))
			volume = volume/255
			volume = torch.from_numpy(volume).float().cuda()
			#print(volume.shape)

			batch = { 'volume': volume }
			k = 0
			c = 0
			with torch.no_grad():
				with torch.cuda.amp.autocast(enabled=True):
					if CFG.tta_rotate == False:
						output =  data_parallel(net,batch)  #net(batch)
						k += output['ink'].data.cpu().numpy()
						c += 1

					#--
					else:
						v = [
							volume,
							torch.rot90(volume, k=1, dims=(-2, -1)),
							torch.rot90(volume, k=2, dims=(-2, -1)),
							torch.rot90(volume, k=3, dims=(-2, -1)),
						]
						K=len(v)
						batch = {
							'volume': torch.cat(v,0)
						}
						output = data_parallel(net,batch)  # net(batch)
						ink = output['ink']

						B,_,h,w = volume.shape
						ink = ink.reshape(K, B, 1, h, w)
						ink = [
							ink[0],
							torch.rot90(ink[1], k=-1, dims=(-2, -1)),
							torch.rot90(ink[2], k=-2, dims=(-2, -1)),
							torch.rot90(ink[3], k=-3, dims=(-2, -1)),
							#torch.rot90(ink[0], k=-1, dims=(-2, -1)),
						]
						ink = torch.stack(ink, dim=0)
						ink = ink.mean(0)

						k += ink.data.cpu().numpy()
						c += 1

			k = k/c
			##print(k.shape)
			B = len(k)
			for b in range(B):
				x0,y0 = xy0[b]
				scale_probability[y0:y0 + crop_size, x0:x0 + crop_size] += k[b,0]*infer_mask
				scale_count[y0:y0 + crop_size, x0:x0 + crop_size] += infer_mask

		##-----------------
		scale_probability = scale_probability/(scale_count+0.000001)
		scale_probability = scale_probability[:sH,:sW]

		p = cv2.resize(scale_probability, dsize=(W,H))
		probability = probability+p
		count = count+1

	#######################################################

	print('')
	probability = probability/(count+0.000001)
	#probability = probability[:H,:W]
	probability = probability*d.mask
	return probability




#################################################################################################


def run_local_cv():

	net = Net(CFG)
	print(checkpoint)
	f = torch.load(checkpoint, map_location=lambda storage, loc: storage)
	state_dict = f['state_dict']
	print(net.load_state_dict(state_dict, strict=True))  # True


	z0 = CFG.read_fragment_z[0]
	z1 = CFG.read_fragment_z[1]
	d = read_crop_data(fold, z0,z1)

	probability = infer_one_ms(net, d)

	#--- draw results
	probability8 = (probability * 255).astype(np.uint8)
	overlay8 = draw_contour_overlay(cv2.cvtColor(probability8, cv2.COLOR_GRAY2BGR), d.label, color=(0, 0, 255), thickness=10)
	image_show_norm(f'overlay', overlay8, resize=0.1)
	image_show_norm(f'probabbility', probability8, resize=0.1)
	image_show_norm(f'label', d.label, resize=0.1)

	cv2.imwrite(f'{infer_dir}/probability8.png',probability8)
	cv2.imwrite(f'{infer_dir}/overlay8.png',overlay8)

	print('metric_to_text() ...')
	t = metric_to_text(probability, d.label, d.mask)
	log.write(f'infer_dir={infer_dir}\n')
	log.write(f'model_path={checkpoint}\n')
	log.write(f'fragment_id={d.fragment_id}\n')
	log.write(cfg_to_text())
	log.write(f'\n')
	log.write(t)
	log.write(f'\n')
	cv2.waitKey(0)

# main #################################################################
if __name__ == '__main__':
	run_local_cv()
