from common import *
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score

#----------------

#https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/397279
#https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/395090

# fbeta lb metric
def compute_lb_score(predict, truth):
	p = predict.reshape(-1)
	t = truth.reshape(-1)

	tp = p*t
	#fp = p*(1-t)
	#fn = (1-p)*t
	prec   = tp.sum()/(p.sum()+0.0001)
	recall = tp.sum()/(t.sum()+0.0001)

	beta=0.5
	#  0.2*1/recall + 0.8*1/prec
	score = beta*beta/(1+beta*beta)*1/recall + 1/(1+beta*beta)*1/prec
	score = 1/score

	#score = fbeta_score(t, p, beta=0.5)
	return score, prec, recall

def mask_to_inner_contour(mask):
	mask = mask>0.5
	pad = np.lib.pad(mask, ((1, 1), (1, 1)), 'reflect')
	contour = mask & (
			(pad[1:-1,1:-1] != pad[:-2,1:-1]) \
			| (pad[1:-1,1:-1] != pad[2:,1:-1]) \
			| (pad[1:-1,1:-1] != pad[1:-1,:-2]) \
			| (pad[1:-1,1:-1] != pad[1:-1,2:])
	)
	return contour

def draw_contour_overlay(image, mask, color=(0,0,255), thickness=1):
	contour =  mask_to_inner_contour(mask)
	if thickness==1:
		image[contour] = color
	else:
		r = max(1,thickness//2)
		for y,x in np.stack(np.where(contour)).T:
			cv2.circle(image, (x,y), r, color, lineType=cv2.LINE_4 )
	return image

# main #################################################################
if __name__ == '__main__': 
	pass
