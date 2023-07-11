from common import *
from ok.model.daformer import *
from ok.model.pvt_v2_2 import *
from einops import rearrange, reduce, repeat

#######################################################################################

def make_mask(cfg):
	s = cfg.crop_size
	f = cfg.crop_fade
	x = np.linspace(-1, 1, s)
	y = np.linspace(-1, 1, s)
	xx, yy = np.meshgrid(x, y)
	d = 1 - np.maximum(np.abs(xx), np.abs(yy))
	d1 = np.clip(d, 0, f / s * 2)
	d1 = d1 / d1.max()
	mask = d1
	return mask


#######################################################################################

class Net(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.output_type = ['inference', 'loss']

		# --------------------------------
		self.cfg = cfg
		self.loss_mask = make_mask(cfg)
		self.d = 5
		encoder_dim = [64, 128, 320, 512]
		decoder_dim = 256

		encoder  = pvt_v2_b3()
		pretrain = torch.load(f'{PRETRAIN_DIR}/{encoder.pretrain}', map_location=lambda storage, loc: storage)
		print('load_state_dict():', encoder.load_state_dict(pretrain, strict=True))

		# b0: 32 #b1: 64
		if 1:
			weight = encoder.patch_embed1.proj.weight.clone()
			weight = torch.cat([weight]*int(np.ceil(self.d/3)),1)
			encoder.patch_embed1.proj = nn.Conv2d(self.d, 64, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))
			weight = encoder.patch_embed1.proj.weight.data[...] = weight[:,:self.d]

		#----
		self.encoder = encoder
		self.decoder = DaformerDecoder(
				encoder_dim=encoder_dim,
				decoder_dim=decoder_dim,
				fuse='conv1x1',
				dilation=None, #[1, 6, 12, 18],
		)
		self.logit = nn.Conv2d(decoder_dim, 1, kernel_size=1)


	def forward(self, batch):
		v = batch['volume']
		B, C, H, W = v.shape
		vv = [
			v[:, i:i + self.d] for i in range(0,C-self.d+1,2)
		]
		K = len(vv)
		x = torch.cat(vv, 0)

		# ---------------------------------
		encoder = self.encoder(x)
		for i in range(len(encoder)):
			e = encoder[i]
			_, c, h, w = e.shape
			e = rearrange(e, '(K B) c h w -> K B c h w', K=K, B=B, h=h, w=w)  #
			m1 = e.mean(0)
			encoder[i] = m1
		###[print(f'encoder1_{i}',f.shape) for i,f in enumerate(encoder)]

		last, decoder = self.decoder(encoder)
		###print('last', last.shape)

		# ---------------------------------
		last = F.dropout(last, p=0.5, training=self.training)
		logit = self.logit(last)
		###print('logit', logit.shape, H/logit.shape[2])


		output = {}
		if 'loss' in self.output_type:
			# output['label_loss'] = F.binary_cross_entropy_with_logits(logit, batch['label'])
			output['label_loss'] = criterion_label_loss(logit, batch['label'], batch['valid'],self.loss_mask)

		if 'inference' in self.output_type:
			if logit.shape[2:]!=(H, W):
				logit = F.interpolate(logit, size=(H, W), mode='bilinear', align_corners=False, antialias=True)
			output['ink'] = torch.sigmoid(logit)

		return output


def criterion_label_loss(logit, truth, valid,loss_mask):
	#if (truth.sum() == 0) : logit = logit.detach() #no gradient

	b, _, lh, lw = logit.shape
	b, _, th, tw = truth.shape
	mask = torch.from_numpy(loss_mask).unsqueeze(0).unsqueeze(0)
	mask = mask.to(truth.device)

	if (lh, lw) != (th, tw):
		truth = F.interpolate(truth, size=[lh, lw], mode='bilinear', align_corners=False)
		truth = (truth > 0.5).float()

		valid = F.interpolate(valid, size=[lh, lw], mode='bilinear', align_corners=False)
		valid = (valid > 0.5).float()

		mask = F.interpolate(mask, size=[lh, lw], mode='bilinear', align_corners=False)

	mask = mask.repeat(b, 1, 1, 1)
	m    = mask*valid
	m    = m.reshape(b, -1)
	t    = truth.reshape(b, -1)
	l    = logit.reshape(b, -1)
	pos  = F.logsigmoid(l)
	neg  = F.logsigmoid(-l)

	loss = - t * pos - (1 - t) * neg
	loss = (loss * m).sum() / m.sum()

	# loss  = F.binary_cross_entropy_with_logits(logit, truth)
	return loss



def run_check_net():
	cfg = dotdict(
		crop_fade=32,
		crop_size=224,
		crop_depth=32,
	)
	height, width =  cfg.crop_size, cfg.crop_size
	depth = cfg.crop_depth
	batch_size = 3

	batch = {
		'volume': torch.from_numpy(np.random.choice(256, (batch_size, depth, height, width))).cuda().float(),
		'label': torch.from_numpy(np.random.choice(2, (batch_size, 1, height, width))).cuda().float(),
		'valid': torch.from_numpy(np.random.choice(2, (batch_size, 1, height, width))).cuda().float(),
	}

	net = Net(cfg).cuda()
	# print(net)
	while 1:
		with torch.no_grad():
			with torch.cuda.amp.autocast(enabled=True):
				print('running')
				output = net(batch)
				break

	# ---

	print('batch')
	for k, v in batch.items():
		print(f'{k:>32} : {v.shape} ')

	print('output')
	for k, v in output.items():
		if 'loss' not in k:
			print(f'{k:>32} : {v.shape} ')
	print('loss')
	for k, v in output.items():
		if 'loss' in k:
			print(f'{k:>32} : {v.item()} ')


# main #################################################################
if __name__ == '__main__':
	run_check_net()





