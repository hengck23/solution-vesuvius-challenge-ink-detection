from common import *
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder, DecoderBlock
from timm.models.resnet import *
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
class SmpUnetDecoder(nn.Module):
	def __init__(self,
				 in_channel,
				 skip_channel,
				 out_channel,
				 ):
		super().__init__()
		self.center = nn.Identity()

		i_channel = [in_channel, ] + out_channel[:-1]
		s_channel = skip_channel
		o_channel = out_channel
		block = [
			DecoderBlock(i, s, o, use_batchnorm=True, attention_type=None)
			for i, s, o in zip(i_channel, s_channel, o_channel)
		]
		self.block = nn.ModuleList(block)

	def forward(self, feature, skip):
		d = self.center(feature)
		decode = []
		for i, block in enumerate(self.block):
			# print(i, d.shape, skip[i].shape if skip[i] is not None else 'none')
			# print(block.conv1[0])
			# print('')

			s = skip[i]
			d = block(d, s)
			decode.append(d)

		last = d
		return last, decode


#######################################################################################

class Net(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.output_type = ['inference', 'loss']

		# --------------------------------
		self.cfg = cfg
		self.loss_mask = make_mask(cfg)
		self.d = 5

		conv_dim = 64
		encoder_dim = [conv_dim, 64, 128, 256, 512, ]
		decoder_dim = [256, 128, 64, 32, 16 ]

		self.encoder = resnet34d(pretrained=True, in_chans=self.d)

		self.decoder = SmpUnetDecoder(
			in_channel=encoder_dim[-1],
			skip_channel=encoder_dim[:-1][::-1]+[0],
			out_channel=decoder_dim,
		)
		self.logit = nn.Conv2d(decoder_dim[-1], 1, kernel_size=1)



	def forward(self, batch):
		v = batch['volume']
		B, C, H, W = v.shape
		vv = [
			v[:, i:i + self.d] for i in range(0,C-self.d+1,2)
		]
		K = len(vv)
		x = torch.cat(vv, 0)
		#print(K)

		# ---------------------------------
		encoder = []
		e = self.encoder

		x = e.conv1(x)
		x = e.bn1(x)
		x = e.act1(x); encoder.append(x)
		x = F.avg_pool2d(x, kernel_size=2, stride=2)
		x = e.layer1(x); encoder.append(x)
		x = e.layer2(x); encoder.append(x)
		x = e.layer3(x); encoder.append(x)
		x = e.layer4(x); encoder.append(x)
		### [print(f'encoder1_{i}',f.shape) for i,f in enumerate(encoder)]

		for i in range(len(encoder)):
			e = encoder[i]
			_, c, h, w = e.shape
			e = rearrange(e, '(K B) c h w -> B K c h w', K=K, B=B, h=h, w=w)  #
			e = e.mean(1)
			encoder[i] = e

		last, decoder = self.decoder(feature=encoder[-1], skip= encoder[:-1][::-1]+[None])
		### [print(f'decoder1_{i}',f.shape) for i,f in enumerate(decoder)]
		###print('last', last.shape)

		# ---------------------------------
		last = F.dropout(last, p=0.5, training=self.training)
		logit = self.logit(last)


		output = {}
		if 'loss' in self.output_type:
			# output['label_loss'] = F.binary_cross_entropy_with_logits(logit, batch['label'])
			output['label_loss'] = criterion_label_loss(logit, batch['label'],self.loss_mask)

		if 'inference' in self.output_type:
			if logit.shape[2:]!=(H, W):
				logit = F.interpolate(logit, size=(H, W), mode='bilinear', align_corners=False, antialias=True)
			output['ink'] = torch.sigmoid(logit)

		return output




def criterion_label_loss(logit, truth, loss_mask):
	#if (truth.sum() == 0) : logit = logit.detach() #no gradient

	b, _, lh, lw = logit.shape
	b, _, th, tw = truth.shape
	mask = torch.from_numpy(loss_mask).unsqueeze(0).unsqueeze(0)
	mask = mask.to(truth.device)

	if (lh, lw) != (th, tw):
		truth = F.interpolate(truth, size=[lh, lw], mode='bilinear', align_corners=False)
		truth = (truth > 0.5).float()
		mask = F.interpolate(mask, size=[lh, lw], mode='bilinear', align_corners=False)

	mask = mask.repeat(b, 1, 1, 1)
	m = mask.reshape(b, -1)
	t = truth.reshape(b, -1)
	l = logit.reshape(b, -1)
	pos = F.logsigmoid(l)
	neg = F.logsigmoid(-l)

	loss = -t * pos - (1 - t) * neg
	loss = (loss * m).sum() / m.sum()
	# loss  = F.binary_cross_entropy_with_logits(logit, truth)
	return loss


def run_check_net():
	cfg = dotdict(
		crop_fade=32,
		crop_size=128,
		crop_depth=32,
	)
	height, width =  cfg.crop_size, cfg.crop_size
	depth = cfg.crop_depth
	batch_size = 3

	batch = {
		'volume': torch.from_numpy(np.random.choice(256, (batch_size, depth, height, width))).cuda().float(),
		'label': torch.from_numpy(np.random.choice(2, (batch_size, 1, height, width))).cuda().float(),
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





