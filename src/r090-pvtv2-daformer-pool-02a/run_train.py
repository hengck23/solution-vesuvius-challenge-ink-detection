import sys, os
import importlib
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
	module = importlib.import_module(sys.argv[1])
	# module = importlib.import_module('config_fold1_stage1_0')
except:
	print('ERROR in sys.argv[1] !!!!!!')
	print('  please use: python run_train.py <config_file.py>')
	print('         e.g: python run_train.py config_fold1_stage1_0')
	exit(0)


#------------------------------------------
# from config_fold1_stage1_0 import *
# from config_fold1_stage1_1 import *
# from config_fold1_stage2_0 import *
#
# from config_fold2aa_stage1_0 import *
# from config_fold2aa_stage2_0 import *
#------------------------------------------

#https://stackoverflow.com/questions/21221358/python-how-to-import-all-methods-and-attributes-from-a-module-dynamically
module_dict = module.__dict__

try:
    to_import = module_dict.__all__
except AttributeError:
    to_import = [name for name in module_dict if not name.startswith('_')]
globals().update({name: module_dict[name] for name in to_import})

######################################################################################################################
os.environ['CUDA_VISIBLE_DEVICES']='0,1'

from common import *
from my_lib.net.lookahead import *
from my_lib.net.rate import *
from dataset import *
from model import *
from kaggle_ink_v1 import *


def infer_one(net, d):
	infer_mask = make_mask(CFG)

	net = net.cuda()
	net = net.eval()
	net.output_type = ['inference']

	#get coord
	crop_size = CFG.crop_size
	stride = crop_size//2
	H,W,D  = d.volume.shape

	##pad #assume H,W >size
	px,py = W%stride,H%stride
	if (px!=0) or (py!=0):
		px = stride-px
		py = stride-py
		pad_volume = np.pad(d.volume, [(0, py), (0, px), (0, 0)], constant_values=0)
	else:
		pad_volume = d.volume

	pH, pW, _  = pad_volume.shape
	x = np.arange(0,pW-crop_size+1,stride)
	y = np.arange(0,pH-crop_size+1,stride)
	x,y = np.meshgrid(x,y)
	xy  = np.stack([x,y],-1).reshape(-1,2)
	#print('H,W,pH,pW,len(xy)',H,W,pH,pW,len(xy))

	probability = np.zeros((pH,pW))
	count = np.zeros((pH,pW))

	start_timer = timer()
	batch_iter = np.array_split(xy, len(xy)//4)
	for t, xy0 in enumerate(batch_iter):
		print('\r @infer_one():', f'crop_id={d.crop_id}', t, len(batch_iter), time_to_str(timer() - start_timer,'sec'), end='')

		#https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/409770
		volume =[]
		for x0,y0 in xy0 :
			v = pad_volume[y0:y0 + crop_size, x0:x0 + crop_size]
			volume.append(v)
		volume = np.stack(volume)
		volume = np.ascontiguousarray(volume.transpose(0,3,1,2))
		volume = volume/255
		volume = torch.from_numpy(volume).float().cuda()
		##print(volume.shape)

		batch = { 'volume': volume }
		k = 0
		c = 0
		with torch.no_grad():
			with torch.cuda.amp.autocast(enabled=False):
				output =  data_parallel(net,batch)#net(batch)
				k += output['ink'].data.cpu().numpy()
				c += 1
		k = k/c
		##print(k.shape)

		batch_size = len(k)
		for b in range(batch_size):
			x0,y0 = xy0[b]
			probability[y0:y0 + crop_size, x0:x0 + crop_size] += k[b,0]*infer_mask
			count[y0:y0 + crop_size, x0:x0 + crop_size] += infer_mask

	#print('')
	probability = probability/(count+0.000001)
	probability = probability[:H,:W]
	probability = probability*d.mask
	return probability



def compute_metric(ink, label, mask):

	p = ink[mask>0.5].reshape(-1)
	t = label[mask>0.5].reshape(-1)
	pos = np.log(np.clip(p,1e-7,1))
	neg = np.log(np.clip(1-p,1e-7,1))
	bce = -(t*pos +(1-t)*neg).mean()

	mask_sum = mask.sum()
	metric = []
	for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
		p = ink[mask>0.5].reshape(-1)
		t = label[mask>0.5].reshape(-1)
		p = (p > threshold).astype(np.float32)
		t = (t > 0.5).astype(np.float32)

		tp = p * t
		precision = tp.sum() / (p.sum() + 0.0001)
		recall = tp.sum() / t.sum()

		fp = p * (1 - t)
		fpr = fp.sum() / (1 - t).sum()

		beta = 0.5
		score = beta * beta / (1 + beta * beta) / (recall+0.0001) + 1 / (1 + beta * beta) / (precision+0.00001)
		score = 1 / score

		dice = 2 * tp.sum() / (p.sum() + t.sum())
		p_sum = p.sum()/mask_sum

		metric.append([threshold, p_sum, precision, recall, fpr,  dice, score])

	metric = np.array(metric)
	return bce, metric

#################################################################################################

def do_valid(net, valid_data, iteration):
	d = valid_data[0]
	probability = infer_one(net, d)
	label = d.label

	#---
	if 1: #debug and save intermediate result
		label8 = (label * 255).astype(np.uint8)
		probability8 = (probability * 255).astype(np.uint8)
		overlay1 = draw_contour_overlay(cv2.cvtColor(probability8, cv2.COLOR_GRAY2BGR), label, color=(0, 0, 255), thickness=10)

		predict = probability>0.5
		predict8 = (predict * 255).astype(np.uint8)
		overlay2 = overlay1.copy()
		overlay2[predict]=[255,255,0]
		overlay2 = draw_contour_overlay(overlay2, label, color=(0, 0, 255), thickness=10)

		resize=0.1
		image_show_norm(f'valid: overlay1', overlay1, resize=resize)
		image_show_norm(f'valid: overlay2', overlay2, resize=resize)
		image_show_norm(f'valid: probability', probability8, resize=resize)
		image_show_norm(f'valid: label', label, resize=resize)
		cv2.waitKey(1)

		cv2.imwrite(f'{fold_dir}/valid/more/{iteration}.probability.png', probability8)
		cv2.imwrite(f'{fold_dir}/valid/more/{iteration}.overlay1.png', overlay1)
		cv2.imwrite(f'{fold_dir}/valid/more/{iteration}.overlay2.png', overlay2)
		cv2.imwrite(f'{fold_dir}/valid/label.png', label8)


	#---
	bce, metric = compute_metric(probability, d.label, d.mask)
	k = np.argmax(metric[:,-1])
	threshold, p_sum, precision, recall, fpr, dice, score = metric[k]

	return [bce, threshold, recall, precision, fpr, p_sum, score]



##----------------
def run_train():
	augment = {
		'train_augment_v2': train_augment_v2,
		'train_augment_v2f': train_augment_v2f,
	}[train_augment]


	## setup  ----------------------------------------
	for f in ['checkpoint','train','valid','valid/more','backup'] : os.makedirs(fold_dir +'/'+f, exist_ok=True)

	log = Logger()
	log.open(fold_dir+'/log.train.txt',mode='a')
	log.write(f'\n--- [START {log.timestamp()}] {"-"*64}\n\n')
	log.write(f'\t{set_environment()}\n')
	log.write(f'\t__file__ = {__file__}\n')
	log.write(f'\tfold_dir = {fold_dir}\n')
	log.write(f'\n')
	log.write(cfg_to_text())
	log.write(f'\n\n')

	## dataset ----------------------------------------
	log.write('** dataset setting **\n')

	train_data, valid_data = make_fold(fold, CFG)
	train_dataset = InkDataset(train_data,augment=augment,cfg=CFG)
	train_loader  = DataLoader(
		train_dataset,
		sampler = RandomSampler(train_dataset),
		batch_size  = batch_size,
		drop_last   = True,
		num_workers = 32,
		pin_memory  = True,
		worker_init_fn = lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
		collate_fn = null_collate,
	)
	log.write(f'fold = {fold}\n')
	log.write(f'train_dataset : \n{str(train_dataset)}\n')
	log.write(f'train_dataset.augment : \n{str(train_dataset.augment)}\n')
	log.write('\n')

	## net ----------------------------------------
	log.write('** net setting **\n')

	scaler = torch.cuda.amp.GradScaler(enabled = True)
	net = Net(CFG)

	if initial_checkpoint is not None:
		f = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
		start_iteration = f.get('iteration',0)
		start_epoch = f.get('epoch',0)
		state_dict = f['state_dict']

		net_state_dict = net.state_dict()
		key = list(state_dict.keys())
		for k in key:
			if state_dict[k].shape != net_state_dict[k].shape:
				print(f'delete {k}')
				del state_dict[k]

		print(net.load_state_dict(state_dict,strict=False))  #True
	else:
		start_iteration = 0
		start_epoch = 0

	print('start_iteration',start_iteration)
	print('start_epoch',start_epoch)

	net.cuda()
	log.write(f'\tinitial_checkpoint = {initial_checkpoint}\n')
	log.write(f'\n')


	## optimiser ----------------------------------
	if is_freeze_encoder:
		for p in net.encoder.parameters():   p.requires_grad = False

	optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()),lr=start_lr)
	#optimizer = Lookahead(RAdam(filter(lambda p: p.requires_grad, net.parameters()),lr=start_lr), alpha=0.5, k=5)

	log.write('optimizer\n  %s\n'%(optimizer))
	log.write('\n')

	num_iteration = (num_epoch-start_epoch+0.01)*len(train_loader) + start_iteration
	iter_log   = len(train_loader) *1
	iter_valid = iter_log
	iter_save  = iter_log
 
	## start training here! ##############################################
	log.write('** start training here! **\n')
	log.write('   batch_size = %d \n'%(batch_size))
	log.write('   experiment = %s\n' % str(__file__.split('/')[-2:]))
	log.write('                           |----------------------- VALID----------------------|---- TRAIN/BATCH ----------------------\n')
	log.write('rate      iter       epoch | loss   thr   recall  prec    fpr   p_sum  score   | loss                 | time           \n')
	log.write('-----------------------------------------------------------------------------------------------------------------------\n')
			  #1.00e-4   00012000*  10.00 | 2.382  0.464  0.593  0.727  0.727  0.727  0.727   | 1.658  0.000  0.000  |  0 hr 15 min

	def message(mode='print'):
		asterisk = ' '
		if mode==('print'):
			loss = batch_loss
		if mode==('log'):
			loss = train_loss
			if (iteration % iter_save == 0): asterisk = '*'

		text = \
			('%0.2e   %08d%s %6.2f | '%(rate, iteration, asterisk, epoch,)).replace('e-0','e-').replace('e+0','e+') + \
			'%4.3f  %4.3f  %4.3f  %4.3f  %4.3f  %4.3f  %4.3f  | '%(*valid_loss,) + \
			'%4.3f  %4.3f  %4.3f  | '%(*loss,) + \
			'%s' % (time_to_str(timer() - start_timer,'min'))

		return text

	#----
	valid_loss = np.zeros(7,np.float32)
	train_loss = np.zeros(3,np.float32)
	batch_loss = np.zeros_like(train_loss)
	sum_train_loss = np.zeros_like(train_loss)
	sum_train = 0


	start_timer = timer()
	iteration = start_iteration
	epoch = start_epoch
	rate = 0

	while iteration < num_iteration:
		for t, batch in enumerate(train_loader):

			if iteration%iter_save==0:
				if iteration != start_iteration:
					skip_save_epoch = 0
					n = iteration if epoch > skip_save_epoch else 0
					torch.save({
						'state_dict': net.state_dict(),
						'iteration': iteration,
						'epoch': epoch,
					}, fold_dir + f'/checkpoint/{n:08d}.model.pth')
					pass


			if (iteration%iter_valid==0): # or (t==len(train_loader)-1):
				if iteration!=start_iteration:
					valid_loss = do_valid(net, valid_data, f'{iteration:08d}')  #
				pass


			if (iteration%iter_log==0) or (iteration%iter_valid==0):
				print('\r', end='', flush=True)
				log.write(message(mode='log') + '\n')


			# learning rate schduler ------------
			# adjust_learning_rate(optimizer, scheduler(epoch))
			rate = get_learning_rate(optimizer)[0]

			# one iteration update  -------------
			B = len(batch['index'])
			batch['volume'] = batch['volume'].cuda()
			batch['label'] = batch['label'].cuda()
			batch['valid'] = batch['valid'].cuda()

			net.train()
			net.output_type = ['loss', 'inference']
			#with torch.autograd.set_detect_anomaly(True):
			if 1:
				with torch.cuda.amp.autocast(enabled = True):
					output = data_parallel(net,batch) #net(batch)#data_parallel(net,batch)
					loss0  = output['label_loss'].mean()

				optimizer.zero_grad()
				scaler.scale(
					  loss0
				).backward()

				#scaler.unscale_(optimizer)
				#torch.nn.utils.clip_grad_norm_(net.parameters(), 2)
				scaler.step(optimizer)
				scaler.update()


			# print statistics  --------
			batch_loss[:3] = [loss0.item(),0,0]#loss1.item(),0] #
			sum_train_loss += batch_loss
			sum_train += 1
			if t % 100 == 0:
				train_loss = sum_train_loss / (sum_train + 1e-12)
				sum_train_loss[...] = 0
				sum_train = 0

			print('\r', end='', flush=True)
			print(message(mode='print'), end='', flush=True)
			epoch += 1 / len(train_loader)
			iteration += 1

			# debug  --------
			if 0:
			# if t % 500 == 0:

				label = batch['label'].float().data.cpu().numpy()
				ink = output['ink'].float().data.cpu().numpy()

				for b in range(len(batch['index'])):
					t = label[b, 0]
					p = ink[b, 0]
					image_show_norm('t', t, min=0, max=1, resize=1)
					image_show_norm('p', p, min=0, max=1, resize=1)


					p8 = (p*255).astype(np.uint8)
					p8 = cv2.cvtColor(p8,cv2.COLOR_GRAY2BGR)
					overlay = draw_contour_overlay(p8, t, color=(0, 0, 255), thickness=10)
					image_show_norm('overlay', overlay, resize=1)
					cv2.waitKey(1)

			# debug  --------

		torch.cuda.empty_cache()
	log.write('\n')

# main #################################################################
if __name__ == '__main__':
	run_train()

