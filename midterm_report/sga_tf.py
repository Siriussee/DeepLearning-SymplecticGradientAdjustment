def jac_vec(ys, xs, vs):
    return fwd_gradients(ys, xs, grad_xs=vs, stop_gradients=xs)

def jac_tran_vec(ys, xs, vs) :
    dydxs = tf.gradients(ys, xs, grad_ys=vs, stop_gradients=xs)
    return [tf.zeros_like(x) if dydx is None else dydx 
        for (x, dydx) in zip(xs,dydxs)]

# return A^T*grad
def get_sym_adj(Ls, xs) :
    xi = [tf.gradients(l, x)[0] for (l, x) in zip(Ls, xs)]
    H_xi = jac_vec(xi, xs, xi)
    Ht_xi = jac_tran_vec(xi, xs, xi)
    At_xi = [(ht−h)/2 for (h, ht) in zip(H_xi, Ht_xi)]
    return At_xi