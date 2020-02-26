r"""
losses l=set(l_i), weights w=set(w_i)
grad <= [gradient(l_i,w_i) for (l_i,w_i) in (l,w)]
A^T*grad <= get_sym_adj(l,w)
output: grad+A^T*grad
"""