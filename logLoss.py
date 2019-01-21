import numpy as np

def compute_log_loss(predicted,actual,eps = 1e-14):
	predicted = np.clip(predicted,eps,1-eps)
	loss = -1* np.mean(actual* np.log(predicted) + (1-actual)*np.log(1-predicted))

	return loss

#example

compute_log_loss(.9,.5)