import tensorflow as _tf

def morse(x, lamb = 0.23314621043800202, alpha = 0.6797779934458726):
    return lamb*(_tf.exp(-2.0*alpha*(x))-2.0*_tf.exp(-alpha*(x))

def stash(x, lamb = 1.1613855392326946, alpha = 0.6520042387583171):
    return _tf.where(x <= 0.0, 2.0*_tf.tanh(alpha*x),
	   lamb*_tf.asinh(2.0*alpha*x/lamb))

def stash_old(x, lamb = 1.1613326990732873, alpha = 0.6521334159737763):
    return _tf.where(x <= 0.0, 2.0*_tf.tanh(alpha*x),
	   lamb*_tf.asinh(2.0*alpha*x/lamb))
