import numpy as np
from scipy import linalg

def cholesky(B):
    L = np.linalg.cholesky(B)
    U = L.T
    return L,U

def solve1(A,B):
    return linalg.eig(A,B)

def solve(A,B):
    L, U = cholesky(B)
    L_i = np.linalg.inv(L)
    S = np.dot(np.dot(L_i, A),L_i.T)
    l = np.linalg.eig(S)
    return l

def remove_zero(B):
    B = B[~np.all(B == 0, axis=1)]
    B = B[:, ~np.all(B == 0, axis=0)]
    return B

def remove_line(A, d_list):
    return np.delete(np.delete(A, d_list, 0), d_list, 1)

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

if __name__=='__main__':
    A=np.array([[9,6,3],[6,13,11],[3,11,35]])
    cholesky(A)