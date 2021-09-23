"""
    Optimization methods to solve the Gromov-Wasserstein problem.
"""

import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import ot
from time import time
import pdb
from scipy.linalg import qr
from scipy.stats import describe


def orth_procrustes(X, Y, drag=None):
    """
        Solves the classical orthogonal procrustes problem, i.e.

                    min_P ||X - YP'||_F subject to P'P=I,

        where X, Y are n x d and P is d x d. (Note that matrices are given as
        rows of observations, as is common in python ML settings, though not in
        Linear Algebra formulation)

    """
    M = X.transpose() @ Y       # Transpose? g is n x m, X.t is d x n
    if drag is not None:
        M += drag
    U, _, Vh = np.linalg.svd(M)   # M is supposedly n * m
    # After transposing, P is m * n, so can right multiply Y
    P = U@Vh
    return P

def gw_imports(use_gpu):
    global bregman, gwggrad, gwloss, cdist
    if use_gpu:
        print('Using GPU in Gromov-Wasserstein computation')
        global cm, pairwiseEuclideanGPU, init_matrix, cosine_distance_gpu, sinkhorn
        import cudamat as cm
        from ot.gpu import bregman
        from ot.gpu.da import pairwiseEuclideanGPU
        from gpu_utils import cdist
        from gromov_gpu import gwggrad, gwloss, init_matrix, sinkhorn
    else:
        print('*NOT* Using GPU in Gromov-Wasserstein computation')
        from ot import bregman
        from ot.gromov import gwggrad, gwloss


def sinkhorn_knopp(a, b, M, reg, init_u=None, init_v=None, numItermax=1000,
    stopThr=1e-9, verbose=False, log=False, **kwargs):
    """
    **** Ported from POT package, modified to accept warm start on u and v
    Solve the entropic regularization optimal transport problem and return the OT matrix

    The function solves the following optimization problem:

    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F + reg\cdot\Omega(\gamma)

        s.t. \gamma 1 = a

             \gamma^T 1= b

             \gamma\geq 0
    where :

    - M is the (ns,nt) metric cost matrix
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are source and target weights (sum to 1)

    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix scaling algorithm as proposed in [2]_


    Parameters
    ----------
    a : np.ndarray (ns,)
        samples weights in the source domain
    b : np.ndarray (nt,) or np.ndarray (nt,nbb)
        samples in the target domain, compute sinkhorn with multiple targets
        and fixed M if b is a matrix (return OT loss + dual variables in log)
    M : np.ndarray (ns,nt)
        loss matrix
    reg : float
        Regularization term >0
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True


    Returns
    -------
    gamma : (ns x nt) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters

    Examples
    --------

    >>> import ot
    >>> a=[.5,.5]
    >>> b=[.5,.5]
    >>> M=[[0.,1.],[1.,0.]]
    >>> ot.sinkhorn(a,b,M,1)
    array([[ 0.36552929,  0.13447071],
           [ 0.13447071,  0.36552929]])


    References
    ----------

    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal Transport, Advances in Neural Information Processing Systems (NIPS) 26, 2013


    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT

    """

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    if len(a) == 0:
        a = np.ones((M.shape[0],), dtype=np.float64) / M.shape[0]
    if len(b) == 0:
        b = np.ones((M.shape[1],), dtype=np.float64) / M.shape[1]

    # init data
    Nini = len(a)
    Nfin = len(b)

    if len(b.shape) > 1:
        nbb = b.shape[1]
    else:
        nbb = 0

    if log:
        log = {'err': [], 'failed': False}  # Failed will work as an exit code

    # we assume that no distances are null except those of the diagonal of distances
    if (init_u is not None) and (init_v is not None):
        u = init_u
        v = init_v
    else:  # Usual uniform init
        # we assume that no distances are null except those of the diagonal of
        # distances
        if nbb:
            u = np.ones((Nini, nbb)) / Nini
            v = np.ones((Nfin, nbb)) / Nfin
        else:
            u = np.ones(Nini) / Nini
            v = np.ones(Nfin) / Nfin
    uprev = np.zeros(Nini)
    vprev = np.zeros(Nini)
    K = np.exp(-M / reg)
    Kp = (1 / a).reshape(-1, 1) * K
    it = 0
    err = 1
    while (err > stopThr and it < numItermax):
        uprev = u
        vprev = v
        KtransposeU = np.dot(K.T, u)
        v = np.divide(b, KtransposeU)
        u = 1. / np.dot(Kp, v)

        if (np.any(KtransposeU == 0) or
                np.any(np.isnan(u)) or np.any(np.isnan(v)) or
                np.any(np.isinf(u)) or np.any(np.isinf(v))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration', it)
            u = uprev
            v = vprev
            log['failed'] = True
            break
        if it % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            if nbb:
                err = np.sum((u - uprev)**2) / np.sum((u)**2) + \
                    np.sum((v - vprev)**2) / np.sum((v)**2)
            else:
                transp = u.reshape(-1, 1) * (K * v)
                err = np.linalg.norm((np.sum(transp, axis=0) - b))**2
            if log:
                log['err'].append(err)

            if verbose:
                if it % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(it, err))
        it = it + 1
    if log:
        log['u'] = u
        log['v'] = v
        log['it'] = it
    if nbb:  # return only loss
        res = np.zeros((nbb))
        for i in range(nbb):
            res[i] = np.sum(
                u[:, i].reshape((-1, 1)) * K * v[:, i].reshape((1, -1)) * M)
        if log:
            return res, log
        else:
            return res

    else:  # return OT matrix

        if log:
            return u.reshape((-1, 1)) * K * v.reshape((1, -1)), log
        else:
            return u.reshape((-1, 1)) * K * v.reshape((1, -1))


class gromov_wass_solver():
    """
        Gromov Wasserstein Solver Class - main method is from POT, has some extra
        useful functions:
            - use warm starts for inner sinkhorn
            - to compute distances
            - to track progress
    """

    def __init__(self, metric='euclidean', normalize_vecs=False, normalize_dists='mean',
                 loss_fun='square_loss', entropic=True, entreg=1e-1, tol=1e-9,
                 round_g=False, compute_accuracy=True, gpu=False, **kwargs):
        self.metric = metric
        #self.normalize_vecs   = normalize_vecs
        self.normalize_dists = normalize_dists
        self.entropic = entropic
        self.loss_fun = loss_fun
        self.reg = entreg
        self.tol = tol
        self.history = []
        self.compute_accuracy = compute_accuracy
        self.ot_warm = True
        self.gpu = gpu
        # FIXME: Not very pythonic to import here. Not sure how to do otherwise.
        gw_imports(self.gpu)

    def gromov_iter_plot(self, t, C1, C2, G, save_path=None):
        if self.gpu:
            C1 = C1.asarray()
            C2 = C2.asarray()
            G = G.asarray()
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        im0 = ax[0].imshow(C1, cmap='jet')
        ax[0].set_title('Ground Cost Source')
        fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
        im1 = ax[1].imshow(C2, cmap='jet')
        ax[1].set_title('Ground Cost Target')
        fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
        im2 = ax[2].imshow(G, cmap='jet')
        ax[2].set_title('Gamma t={}'.format(t))
        fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
        fig.tight_layout()
        if save_path:
            outpath = os.path.join(save_path, 'gw_iter_' + str(t) + '.pdf')
            plt.savefig(outpath, bbox_inches='tight', format='pdf', dpi=300)
        # plt.show(block=False)
        # plt.draw()

    def print_header(self):
        if not self.compute_accuracy:
            w = 60
            head = '{:>5} {:>10} {:>10} {:>10} {:>8} {:>8}'.format(
                'T', 'obj', 'entropy', 'delta_G', 'time_G', 'time_total')
        else:
            w = 90
            head = '{:>5} {:>10} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8}'.format(
                'T', 'obj', 'entropy', 'delta_G', 'Acc@1', 'Acc@10', 'time_G', 'time_total')
        print('-' * w)
        print(head)
        print('-' * w)

    def print_step(self, it, dist, ent_t, err, time_G, time_total, accs = None):
        if self.compute_accuracy and accs is not None:
            row = '{:5} {:10.2e} {:10.2e} {:10.2e} {:10.2f} {:10.2f} {:8.4f} {:8.4f}' .format(
            it, dist, ent_t, err, 100*accs[1], 100*accs[10], time_G, time_total)
        else:
            row = '{:5} {:10.2e} {:10.2e} {:10.2e} {:8.4f} {:8.4f}' .format(
            it, dist, ent_t, err, time_G, time_total)
        print(row)

    def plot_history(self, save_path=None):
        if self.history is []:
            print("Error: no history")
            return -1
        It, D, E, DG, ACC1, ACC5, ACC10, T = list(
            zip(*self.history))
        fig, ax = plt.subplots(1, 4, figsize=(16, 4))
        ax[0].plot(It, D)
        ax[0].set_xlabel('Iteration')
        ax[0].set_ylabel('Distance')

        ax[1].plot(It, DG, label='delta G')
        ax[1].set_xlabel('Iteration')
        ax[1].set_ylabel('Relative Change (%)')

        ax[2].plot(It, E)
        ax[2].set_xlabel('Iteration')
        ax[2].set_ylabel('(Neg) Entropy')

        ax[3].plot(It, ACC1, label='Acc@1')
        ax[3].plot(It, ACC5, label='Acc@5')
        ax[3].plot(It, ACC10, label='Acc@10')
        ax[3].set_xlabel('Iteration')
        ax[3].set_ylabel('Accuracy (%)')
        ax[3].legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', format='pdf', dpi=300)
        plt.show(block=False)

    def compute_distances(self, X, Y):
        print('Computing intra-domain distance matrices...')

        if not self.gpu:
            C1 = sp.spatial.distance.cdist(X, X, metric=self.metric)
            C2 = sp.spatial.distance.cdist(Y, Y, metric=self.metric)
            if self.normalize_dists == 'max':
                print('here')
                C1 /= C1.max()
                C2 /= C2.max()
            elif self.normalize_dists == 'mean':
                C1 /= C1.mean()
                C2 /= C2.mean()
            elif self.normalize_dists == 'median':
                C1 /= np.median(C1)
                C2 /= np.median(C2)
        else:
            C1 = cdist(X, X, metric=self.metric, returnAsGPU=True)
            C2 = cdist(Y, Y, metric=self.metric, returnAsGPU=True)
            if self.normalize_dists == 'max':
                C1.divide(float(np.max(C1.asarray())))
                C2.divide(float(np.max(C2.asarray())))
            elif self.normalize_dists == 'mean':
                C1.divide(float(np.mean(C1.asarray())))
                C2.divide(float(np.mean(C2.asarray())))
            elif self.normalize_dists == 'median':
                raise NotImplemented(
                    "Median normalization not implemented in GPU yet")

        stats_C1 = describe(C1.flatten())
        stats_C2 = describe(C2.flatten())

        for (k, C, v) in [('C1', C1, stats_C1), ('C2', C2, stats_C2)]:
            print('Stats Distance Matrix {}. mean: {:8.2f}, median: {:8.2f},\
             min: {:8.2f}, max:{:8.2f}'.format(k, v.mean, np.median(C), v.minmax[0], v.minmax[1]))

        self.C1, self.C2 = C1, C2

    def compute_gamma_entropy(self, G):
        if not self.gpu:
            Prod = G * (np.log(G) - 1)
            ent = np.nan_to_num(Prod).sum()
        else:
            Prod = cm.empty(G.shape)
            Prod = G.mult(cm.log(G.copy()).subtract(1), target=Prod)
            ent = np.nan_to_num(Prod.asarray()).sum()
        return ent

    def init_matrix(self, C1, C2, T, p, q, loss_fun='square_loss'):
        """ Return loss matrices and tensors for Gromov-Wasserstein fast computation
        Returns the value of \mathcal{L}(C1,C2) \otimes T with the selected loss
        function as the loss function of Gromow-Wasserstein discrepancy.
        The matrices are computed as described in Proposition 1 in [12]
        Where :
            * C1 : Metric cost matrix in the source space
            * C2 : Metric cost matrix in the target space
            * T : A coupling between those two spaces
        The square-loss function L(a,b)=(1/2)*|a-b|^2 is read as :
            L(a,b) = f1(a)+f2(b)-h1(a)*h2(b) with :
                * f1(a)=(a^2)/2
                * f2(b)=(b^2)/2
                * h1(a)=a
                * h2(b)=b
        The kl-loss function L(a,b)=(1/2)*|a-b|^2 is read as :
            L(a,b) = f1(a)+f2(b)-h1(a)*h2(b) with :
                * f1(a)=a*log(a)-a
                * f2(b)=b
                * h1(a)=a
                * h2(b)=log(b)
        Parameters
        ----------
        C1 : ndarray, shape (ns, ns)
             Metric cost matrix in the source space
        C2 : ndarray, shape (nt, nt)
             Metric costfr matrix in the target space
        T :  ndarray, shape (ns, nt)
             Coupling between source and target spaces
        p : ndarray, shape (ns,)
        Returns
        -------
        constC : ndarray, shape (ns, nt)
               Constant C matrix in Eq. (6)
        hC1 : ndarray, shape (ns, ns)
               h1(C1) matrix in Eq. (6)
        hC2 : ndarray, shape (nt, nt)
               h2(C) matrix in Eq. (6)
        References
        ----------
        .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.
        """

        if loss_fun == 'square_loss':
            def f1(a):
                return (a**2) / 2

            def f2(b):
                return (b**2) / 2

            def h1(a):
                return a

            def h2(b):
                return b
        elif loss_fun == 'kl_loss':
            def f1(a):
                return a * np.log(a + 1e-15) - a

            def f2(b):
                return b

            def h1(a):
                return a

            def h2(b):
                return np.log(b + 1e-15)

        constC1 = np.dot(np.dot(f1(C1), p.reshape(-1, 1)),
                         np.ones(len(q)).reshape(1, -1))
        constC2 = np.dot(np.ones(len(p)).reshape(-1, 1),
                         np.dot(q.reshape(1, -1), f2(C2).T))
        constC = constC1 + constC2
        hC1 = h1(C1)
        hC2 = h2(C2)
        return constC, hC1, hC2

    def solve(self, xs, xt, p, q, maxiter=200, plot_every=20, print_every=1, verbose=True, save_plots=None):
        """
        **** Ported from POT package, modified to plot plans and compute downstream
        error along the way

        Returns the gromov-wasserstein transport between (C1,p) and (C2,q)
        (C1,p) and (C2,q)
        The function solves the following optimization problem:
        .. math::
            \GW = arg\min_T \sum_{i,j,k,l} L(C1_{i,k},C2_{j,l})*T_{i,j}*T_{k,l}-\epsilon(H(T))
            s.t. \GW 1 = p
                 \GW^T 1= q
                 \GW\geq 0
        Where :
            C1 : Metric cost matrix in the source space
            C2 : Metric cost matrix in the target space
            p  : distribution in the source space
            q  : distribution in the target space
            L  : loss function to account for the misfit between the similarity matrices
            H  : entropy
        Parameters
        ----------
        C1 : ndarray, shape (ns, ns)
             Metric cost matrix in the source space
        C2 : ndarray, shape (nt, nt)
             Metric costfr matrix in the target space
        p :  ndarray, shape (ns,)
             distribution in the source space
        q :  ndarray, shape (nt,)
             distribution in the target space
        loss_fun :  string
            loss function used for the solver either 'square_loss' or 'kl_loss'
        epsilon : float
            Regularization term >0
        max_iter : int, optional
           Max number of iterations
        tol : float, optional
            Stop threshold on error (>0)
        verbose : bool, optional
            Print information along iterations
        log : bool, optional
            record log if True
        Returns
        -------
        T : ndarray, shape (ns, nt)
            coupling between the two spaces that minimizes :
                \sum_{i,j,k,l} L(C1_{i,k},C2_{j,l})*T_{i,j}*T_{k,l}-\epsilon(H(T))
        References
        ----------
        .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.
        """
        if self.gpu:
            cm.init()

        self.compute_distances(xs, xt)

        C1 = np.asarray(self.C1, dtype=np.float64)
        C2 = np.asarray(self.C2, dtype=np.float64)
        T = np.outer(p, q)  # Initialization
        if not self.gpu:
            constC, hC1, hC2 = self.init_matrix(C1, C2, T, p, q, self.loss_fun)
        else:
            T = cm.CUDAMatrix(T)
            constC, hC1, hC2 = gromov_gpu.init_matrix(
                C1, C2, T, p, q, self.loss_fun)

        it = 0
        err = 1
        global_start = time()
        # plt.ion() # To plot interactively during training

        while (err > self.tol and it <= maxiter):
            start = time()
            if verbose and (it % plot_every == 0):
                self.gromov_iter_plot(it, C1, C2, T, save_path=save_plots)
                # self.print_header()
            Tprev = T

            # compute the gradient
            tens = gwggrad(constC, hC1, hC2, T)

            # FIXME: Clean this up. Global vars and tailored imports should allow
            # to not have devide gpu and non gpu cases.
            if not self.gpu:
                if self.ot_warm and it > 0:
                    T, log = sinkhorn_knopp(
                        p, q, tens, self.reg, init_u=log['u'], init_v=log['v'], log=True)
                elif self.ot_warm:
                    T, log = bregman.sinkhorn(p, q, tens, self.reg, log=True)
                else:
                    T = bregman.sinkhorn(p, q, tens, self.reg)
            else:
                if self.ot_warm and it > 0:
                    T, log = bregman.sinkhorn(
                        p, q, tens, self.reg, returnAsGPU=True, log=True)
                elif self.ot_warm:
                    T, log = bregman.sinkhorn(p, q, tens, self.reg, log=True)
                else:
                    T = bregman.sinkhorn(
                        p, q, tens, self.reg, returnAsGPU=True)

            time_G = time() - start

            if it % print_every == 0:
                # we can speed up the process by checking for the error only all
                # the 10th iterations
                dist = gwloss(constC, hC1, hC2, T)
                if self.gpu:
                    err = np.linalg.norm(T.copy().subtract(Tprev).asarray())
                else:
                    err = np.linalg.norm(T - Tprev)

                # Debug: Test Accuracy
                if self.compute_accuracy:
                    #accs = self.test_accuracy(G_t)
                    if self.gpu:
                        accs = self.accuracy_function(T.asarray())
                    else:
                        accs = self.accuracy_function(T)
                else:
                    accs = {1: np.nan, 5: np.nan, 10: np.nan}

                ent_t = self.compute_gamma_entropy(T)
                if verbose:
                    if it % 200 == 0:
                        self.print_header()
                    self.print_step(it, dist, ent_t, err, time_G, time() - start, accs = accs)

                self.history.append((it, dist, ent_t, 100 * err,
                                     100 * accs[1], 100 * accs[5], 100 * accs[10], time() - global_start))

            it += 1

        plt.close('all')
        if self.gpu:
            return T.asarray()
        else:
            return T
