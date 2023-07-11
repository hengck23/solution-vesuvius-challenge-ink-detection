from common import *

#------------------------------------------------------------------------------------------------------
fold     = '2aa'
out_dir  = OUT_DIR + '/r050_resnet34-unet-mean32-pool-05'
fold_dir = out_dir + f'/fold-{fold}/stage2_0'

initial_checkpoint = \
	OUT_DIR + f'/r050_resnet34-unet-mean32-pool-05/fold-{fold}/stage1_1/checkpoint/00014122.model.pth'
is_freeze_encoder = True


start_lr = 5e-4
batch_size = 64
num_epoch  = 30

train_augment = 'train_augment_v2'




#------------------------------------------------------------------------------------------------------
class Config(object):
	valid_threshold = 0.80
	beta = 1
	crop_fade  = 32
	crop_size  = 256
	crop_depth = 32
	read_fragment_z = [
		32-16,
		32+16,
	]
	dz = 0

CFG = Config()
CFG.infer_fragment_depth = CFG.read_fragment_z[1] - CFG.read_fragment_z[0]

#print CFG function
def cfg_to_text():
    d = Config.__dict__
    text = [f'\t{k} : {v}' for k,v in d.items() if not (k.startswith('__') and k.endswith('__'))]
    d = CFG.__dict__
    text += [f'\t{k} : {v}' for k,v in d.items() if not (k.startswith('__') and k.endswith('__'))]
    return 'CFG\n'+'\n'.join(text)




# main #################################################################
if __name__ == '__main__':
	print(cfg_to_text(),'\n')





