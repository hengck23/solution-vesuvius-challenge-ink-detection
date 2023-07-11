from common import *

#------------------------------------------------------------------------------------------------------
fold     = '2aa'
out_dir  = OUT_DIR + '/r091_pvt_v2_b3-daformer-mean32-aug2-00'
fold_dir = out_dir + f'/fold-{fold}/stage1_0'

initial_checkpoint = None
is_freeze_encoder = False

start_lr = 1e-4
batch_size = 32
num_epoch  = 25

train_augment = 'train_augment_v2'




class Config(object):
	valid_threshold = 0.70
	beta = 1
	crop_fade  = 32
	crop_size  = 128
	crop_depth = 32
	read_fragment_z = [
		32-16,
		32+16,
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

