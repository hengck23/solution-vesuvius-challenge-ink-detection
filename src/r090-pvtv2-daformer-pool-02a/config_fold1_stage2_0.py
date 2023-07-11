from common import *

#------------------------------------------------------------------------------------------------------
fold     = '1'
out_dir  = OUT_DIR + '/r091_pvt_v2_b3-daformer-mean32-aug2-00'
fold_dir = out_dir + f'/fold-{fold}/stage2_0'

initial_checkpoint = \
	OUT_DIR + f'/r091_pvt_v2_b3-daformer-mean32-aug2-00/fold-{fold}/stage1_1/checkpoint/00028080.model.pth'
is_freeze_encoder = False


start_lr = 1e-4
batch_size = 32
num_epoch  = 50

train_augment = 'train_augment_v2f'




class Config(object):
	valid_threshold = 0.70
	beta = 1
	crop_fade  = 32
	crop_size  = 384
	crop_depth = 16
	read_fragment_z = [
		32-8,
		32+8,
	]
	dz = 0

CFG = Config()
CFG.read_fragment_depth = CFG.read_fragment_z[1] - CFG.read_fragment_z[0]


def cfg_to_text():
	d = Config.__dict__
	text = [f'\t{k} : {v}' for k,v in d.items() if not (k.startswith('__') and k.endswith('__'))]
	d = CFG.__dict__
	text += [f'\t{k} : {v}' for k,v in d.items() if not (k.startswith('__') and k.endswith('__'))]
	return 'CFG\n'+'\n'.join(text)


# main #################################################################
if __name__ == '__main__':
	print(cfg_to_text(),'\n')

