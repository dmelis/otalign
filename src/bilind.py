'''
Tools for Bilingual Lexical Induction
Copyright (C) 2018 David Alvarez-Melis <dalvmel@mit.edu>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

import pdb
import os
import numpy as np
import scipy as sp
import scipy.linalg
import matplotlib.pyplot as plt
import src.embeddings as embeddings

import ot

try:  # test if cudamat installed
    from ot.gpu import bregman
    from gw_optim_gpu import gromov_wass_solver
    #from optim import gromov_wass_solver
except ImportError:
    from src.gw_optim import gromov_wass_solver

from src import orth_procrustes


def pprint_golddict(d, src, tgt):
    for i,vals in d.items():
        print(  '{:20s} <-> {:20s}'.format(src[i],','.join([tgt[i] for i in vals])))

def zipf_init(lang, n):
    # Piantadosi, 2014
    #alpha = 1.13 # These numbers are tailoted to english, maybe get best per language?
    #beta  = 2.73
    if lang == 'en':
        alpha, beta = 1.40, 1.88 #1.13, 2.73
    elif lang == 'fi':
        alpha, beta = 1.17, 0.60
    elif lang == 'fr':
        alpha, beta = 1.71, 2.09
    elif lang == 'de':
        alpha, beta = 1.10, 0.40
    elif lang == 'es':
        alpha, beta = 1.84, 3.81
    else: # Deafult to EN
        alpha, beta = 1.40, 1.88 #1.13, 2.73
    p = np.array([1/((i+1)+beta)**(alpha) for i in range(n)])
    return p/p.sum()


class bilingual_mapping(): #rename ot_bilingual_mapping
    """
        Generic class with a lot of useful methods for bilingual mappings
    """
    def __init__(self, xs, xt, src_lang, trg_lang, src_words, trg_words, src_word2ind, trg_word2ind,
        test_dict = None, metric = 'euclidean', normalize_vecs = 'both',
        normalize_dists = False, score_type = 'coupling', adjust = None, distribs = 'uniform', **kwargs):
        self.src_lang        = src_lang
        self.trg_lang        = trg_lang
        self.src_words       = src_words # Source vocabulary, positions correspond to indices
        self.trg_words       = trg_words # Target vocabulary, positions correspond to indices
        self.xs              = xs # Source embedding
        self.xt              = xt # Target embeddings
        self.src_word2ind    = src_word2ind
        self.trg_word2ind    = trg_word2ind
        self.xs_oov          = None
        self.xt_oov          = None
        self.test_dict   = test_dict
        self.scores          = None # Matching scores
        self.coupling        = None
        self.score_type      = score_type
        self.adjust          = adjust
        self.metric          = metric
        self.normalize_vecs  = normalize_vecs
        self.normalize_dists = normalize_dists
        self.Vs              = len(src_words)
        self.Vt              = len(trg_words)
        self.ns, self.ds     = xs.shape
        self.nt, self.dt     = xt.shape
        self.mapping         = None # Will be updated after optim
        assert self.Vs == self.ns
        assert self.Vt == self.nt

        print('Initializing mapping instance. ')

        # 2. Compute marginal distributions
        if distribs == 'uniform':
            self.p = ot.unif(self.ns)
            self.q = ot.unif(self.nt)
        elif distribs == 'zipf':
            self.p = zipf_init(self.src_lang, self.ns)
            self.q = zipf_init(self.trg_lang, self.nt)
        elif distribs == 'file':
            pdb.set_trace()
        else:
            raise ValueError()

        self.mapping = np.eye(self.xs.shape[1])

        print('Source number of words: {}'.format(self.Vs))
        print('Target number of words: {}'.format(self.Vt))
        print('Source embedding dim: {}'.format(self.ds))
        print('Target embedding dim: {}'.format(self.dt))

    def load_from_file(self, path):
        results = pickle.load(open(path, "rb"))
        BLI.mapping  = results['P']
        BLI.coupling = results['G']

    def load_oov_data(self, xs, xt, src_words, trg_words, src_word2ind,
                      trg_word2ind, src2trg, keep_original = True, normalize = True):
        # THIS MAKES THE CRITICAL ASSUMPTION THAT xs_oov, xt_oov, words, etc,
        # are supersets and in same order of xs xt
        n , m = self.ns, self.nt
        if not keep_original:
            self.xs     = xs[:n,:]
            self.xt     = xt[:m,:]
        self.xt_oov = xt[m:,:]
        self.xs_oov = xs[n:,:]
        self.src_words = src_words
        self.trg_words = trg_words
        self.src_word2ind = src_word2ind
        self.trg_word2ind = trg_word2ind
        self.src2trg      = src2trg
        # Clean previous score computation
        self.scores       = None
        assert self.xs.shape[0] + self.xs_oov.shape[0] == len(src_words)
        assert self.xt.shape[0] + self.xt_oov.shape[0] == len(trg_words)

        if normalize:
            self.normalize_embeddings()

    def _test_accuracy(self, G, P = None, verbose = False):
        """
            Dummy function. It's puropose is to have a callable to
            pass to optimization methods like proc_ot and get a testing error
            in each iteration.
        """
        self.coupling = G
        if P is not None:
            # Gromov might not have this one during training
            self.mapping  = P
        self.compute_scores(self.score_type, adjust = self.adjust, verbose=False)
        accs = self.score_translations(self.test_dict, verbose=False)
        if verbose:
            for k, v in accs.items():
                print('Accuracy @{:2}: {:8.2f}%'.format(k,100*v))
        return accs

    def test_accuracy(self, score_type = None, adjust = None, verbose = False):
        """
            Same as above, but uses the object's attribute G.

            Seems uncessary to have both this and compute_scores. Consider merging.
        """
        if score_type is None:
            score_type = self.score_type
        if adjust is None:
            adjust = self.adjust
        if self.coupling is None:
            raise ValueError('Optimal coupling G has not been computed yet')
        self.compute_scores(score_type, adjust, verbose = verbose > 1)  # adjust = 'csls',)
        accs = self.score_translations(self.test_dict, verbose = verbose > 1)
        if verbose > 0:
            for k, v in accs.items():
                print('Accuracy @{:2}: {:8.2f}%'.format(k,100*v))
        return accs

    def oov_translation_old(self, xs, xt, src_words, trg_words, src2trg, adjust = None, verbose = False):#, metric = 'euclidean'):
        """ Requires mapping """
        assert self.mapping is not None
        scores = -sp.spatial.distance.cdist(xs, xt@self.mapping, metric=self.metric)
        if adjust == 'csls':
            scores = csls(scores, knn = 10)
        # Lazy approach: overwrite everything
        self.xs = xs
        self.xt = xt
        self.src_words = src_words
        self.trg_words = trg_words
        self.src2trg   = src2trg
        self.scores    = scores
        accs = self.score_translations(self.test_dict, verbose = verbose > 1)
        if verbose > 0:
            for k, v in accs.items():
                print('Accuracy @{:2}: {:8.2f}%'.format(k,100*v))
        return accs

    def oov_translation(self, xs, xt, src_words, trg_words, src_word2ind, trg_word2ind,
                            src2trg, adjust = None, verbose = False):#, metric = 'euclidean'):
        """ Requires mapping """
        assert self.mapping is not None
        idx_src = [k for k in src2trg.keys()]
        # Only compute scores for src words we will need (but all trg words, otherwise it's cheating!)
        pdb.set_trace()
        nn, scores = csls_sparse(xs, xt@self.mapping, idx_src, range(xt.shape[0]), knn = 10)
        pdb.set_trace()
        # scores = -sp.spatial.distance.cdist(xs, xt@self.mapping, metric=self.metric)
        # if adjust == 'csls':
        #     scores = csls(scores, knn = 10)
        # Lazy approach: overwrite everything
        self.xs = xs
        self.xt = xt
        self.src_words = src_words
        self.trg_words = trg_words
        self.src_word2ind = src_word2ind
        self.trg_word2ind = trg_word2ind
        self.src2trg   = src2trg
        self.scores    = scores
        accs = self.score_translations(self.test_dict, verbose = verbose > 1)
        if verbose > 0:
            for k, v in accs.items():
                print('Accuracy @{:2}: {:8.2f}%'.format(k,100*v))
        return accs


    def compute_scores(self, score_type, adjust = None, verbose = False):
        """ Exlcusive to OT methods. Given coupling matrix, compute scores that will
            be used to determine word translations. Options:
                - coupling: use directly the GW coupling
                - barycentric: using barycenter transported samples to target domain,
                     compute distaces there and use as scores


                - adjust (str): refinement method [None|csls|isf]

        """
        if score_type == 'coupling':
            scores = self.coupling
        elif score_type == 'barycentric':
            ot_emd = ot.da.EMDTransport()
            ot_emd.xs_ = self.xs
            ot_emd.xt_ = self.xt
            ot_emd.coupling_= self.coupling
            #xs_t = ot_emd.transform(Xs=self.xs) # Maps source samples to target space
            #scores = -sp.spatial.distance.cdist(xs_t, self.xt, metric = self.metric) #FIXME: should this be - dist?
            xt_s = ot_emd.inverse_transform(Xt=self.xt) # Maps target to source space
            scores = -sp.spatial.distance.cdist(self.xs, xt_s, metric = self.metric) #FIXME: should this be - dist?
        elif score_type == 'distance':
            # For baselines that only use distances without OT
            scores = -sp.spatial.distance.cdist(self.xs,self.xt, metric = self.metric)
        elif score_type == 'projected':
            # Uses projection mapping, computes distance in projected space
            scores = -sp.spatial.distance.cdist(self.xs,self.xt@self.mapping.T, metric = self.metric)

        if adjust == 'csls':
            scores = csls(scores, knn = 10)
            #print('here')
        elif adjust == 'isf':
            raise NotImplementedError('Inverted Softmax not implemented yet')

        self.scores = scores

        if verbose:
            plt.figure()
            plt.imshow(scores, cmap='jet')
            plt.colorbar()
            plt.show()

    def export_mapped(self, iov_mode = 'matched', oov_mode = 'projection', outf = None, encoding = 'utf-8', suffix = ''):
        """
            Args:
                - mode: how to map:
                    + proc uses procrustes mapping
                    + barycentrinc
                    + add joint OT map learning?
                - right now oov_mode only mode is projection

            Maps target embeddings, keeps source embeddings as is
        """
        # Check if we're in oov case or not.
        n,m = self.coupling.shape

        oov_src = (self.xs_oov is not None) and (n != self.xs_oov.shape[0])
        oov_trg = (self.xt_oov is not None) and (m != self.xt_oov.shape[0])
        any_oov = oov_src or oov_trg
        if any_oov:
            xs_hat = np.concatenate([self.xs,self.xs_oov])
        else:
            xs_hat = self.xs

        # Get the IIVs
        if any_oov and iov_mode == 'projection' and oov_mode == 'projection':
            # Both are projected
            xt_hat = np.concatenate([self.xt,self.xt_oov])@self.mapping
        else:
            # Will treat IIV and OOVs differently
            if iov_mode == 'barycentric':
                ot_emd = ot.da.EMDTransport()
                ot_emd.xs_ = self.xs
                ot_emd.xt_ = self.xt@self.mapping.T
                ot_emd.coupling_= self.coupling
                #xs_hat = ot_emd.transform(Xs=self.xs)
                xt_hat_iov = ot_emd.inverse_transform(Xt=self.xt)
            elif iov_mode == 'matched':
                # Map initial vectors deterministically by their predicted translation
                xt_hat_iov   = self.xs[self.coupling.argmax(axis=0),:]
            elif iov_mode == 'projection':
                xt_hat_iov   = self.xt@self.mapping.T
            if any_oov:
                xt_hat_oov = self.xt_oov@self.mapping.T
                xt_hat = np.concatenate([xt_hat_iov, xt_hat_oov])
            else:
                xt_hat = xt_hat_iov

        # Use muse instead or in addition to this format?
        file_src = os.path.join(outf, ('vectors-%s.' + suffix + '.txt') % self.src_lang)
        file_trg = os.path.join(outf, ('vectors-%s.' + suffix + '.txt') % self.trg_lang)

        try:
            embeddings.write(self.src_words, xs_hat, open(file_src, "w", encoding=encoding, errors='surrogateescape'))
            embeddings.write(self.trg_words, xt_hat, open(file_trg, "w", encoding=encoding, errors='surrogateescape'))
        except:
           print('Problem writing vectors')

    def dump_translations(self, gold_dict, outf, score_type = 'coupling', adjust = None, verbose = 0):
        """
            predicted: dict srcw: [sorted candidates]
            gold:      dict srcw: [translations]

        """
        with open(outf, 'w') as f:
            oov = set()
            n,m = self.coupling.shape # The actual size of mapping computed, might be smaller that total size of dict
            # 1. Compute scores for desired words
            seeds = [self.src_words[s] for s,t in gold_dict.items() if s < len(self.src_words)]

            if self.scores is None:
                self.compute_scores(score_type, adjust, verbose = verbose > 1)  # adjust = 'csls',)

            topk_predictions, oov = self.generate_translations(words = seeds, candidates = 10)

            for src_idx,tgt_idx in gold_dict.items():
                if src_idx > n or np.all([e>m for e in tgt_idx]):
                    continue
                else:
                    src_word = self.src_words[src_idx]
                    predicted = topk_predictions[src_word]
                    targets   = [self.trg_words[j] for j in tgt_idx]
                    row = '\t'.join([src_word,'||'] + predicted)
                    f.write(row + '\n')

                # #print(predicted, targets)
                # correct_strings = []
                # for k in [1,5,10]:
                #     if set(predicted[:k]).intersection(targets):
                #         correct[k] +=1
                #         correct_strings.append('\u2713')
                #     else:
                #         correct_strings.append('X')


    def score_translations(self, gold_dict, verbose = False):
        """
            predicted: dict srcw: [sorted candidates]
            gold:      dict srcw: [translations]

        """
        oov = set()
        #correct = 0
        n,m = self.scores.shape # The actual size of mapping computed, might be smaller that total size of dict

        assert n == self.xs.shape[0], "Score matrix size does not match number of src vecs"
        assert m == self.xs.shape[0], "Score matrix size does not match number of trg vecs"

        correct = {1:0, 5:0, 10:0}

        # 1. Retrieve scores for desired words
        seeds = [self.src_words[s] for s,t in gold_dict.items() if s < len(self.src_words)]

        topk_predictions, oov = self.generate_translations(words = seeds, candidates = 10)


        if verbose:
            print('@1 @5 @10 {:10} {:30} {:80}'.format('Src','Gold','Top K Predicted'))
            print('-'*80)
            print_row = '{:2} {:2} {:2} {:10} {:30} {:80}'


        for src_idx,tgt_idx in gold_dict.items():
            if src_idx > n or np.all([e>m for e in tgt_idx]):
                oov.add(src_idx)
                continue
            else:
                src_word = self.src_words[src_idx]
                predicted = topk_predictions[src_word]
                targets   = [self.trg_words[j] for j in tgt_idx]

                #print(predicted, targets)
                correct_strings = []
                for k in [1,5,10]:
                    if set(predicted[:k]).intersection(targets):
                        correct[k] +=1
                        correct_strings.append('\u2713')
                    else:
                        correct_strings.append('X')

                if verbose:
                    pred_str =  ','.join(predicted)
                    gold_str = ','.join(targets)
                    print(print_row.format(*correct_strings,src_word,gold_str,pred_str))


        precisions = {k: v / len(gold_dict) for k,v in correct.items()}
        coverage = len(gold_dict.keys()) / (len(gold_dict.keys()) + len(oov))
        if verbose:
            print('-'*80)
            print('Coverage: {:7.2%}  Precisions: @1:{:7.2%} @5:{:7.2%} @10:{:7.2%}'.format(coverage, precisions[1], precisions[5], precisions[10]))
        return  precisions

    def generate_translations(self, words = None, candidates = 1, nn_method = 'naive', return_scores = False):
        """ ~ predict method in scikit classes. For now, it's transductive,
            computes translations for words specified from the beginning.
            TODO: Generalize to predict for new ones too whenever possible,
            analogous to .predict() being used with train or test (new) data

            FIXME: Maybe we shouldn't compute translations for all source words, if
            we expect eval set to have far less. But seems that at least scoring methods should
            incorporate info from all, to normalize properly. -> Another arument in favor
            of separate compute_scores and generate_translations methods.

            If words provided, only compute translation for those.

                - candidates (int): number of translation candidates to return. If >1
                                    returns them in decreasing order of relevance (score)


        """
        oov = set()
        k = candidates
        predicted = {}

        if words is None:
            words = self.src_words

        for ws in words:
            if not ws in self.src_word2ind:
                oov.add(ws)
                continue
            src_idx = self.src_word2ind[ws]

            if nn_method == 'naive':
                # For each word (row), select target words (columns) with largest score - no intra-row agreement
                knn = np.argpartition(self.scores[src_idx,:], -k)[-k:] # argpartition returns top k not in order but it's efficient (doesnt sort all rows)
                knn_sort = knn[np.argsort(-self.scores[src_idx,knn])] # With - to get descending order

            translations = [self.trg_words[k] for k in knn_sort]

            if return_scores:
                translation_scores = self.scores[src_idx,knn_sort]
            predicted[ws] = translations

        return predicted, oov

    def find_mutual_nn(self):
        """ Finds mutual nereast neighbors among the whole source and target words """
        best_match_src = self.scores.argmax(1) # Best match for each source word
        best_match_trg = self.scores.argmax(0) # Best match for each source word

        # ONELIENER
        # paired_idx = [(i,best_match_src[i]) for i in range(self.ns) if best_match_trg[best_match_src[i]] == i]
        # paired_words = [(self.src_words[i],self.trg_words[j]) for (i,j) in paired_idx]
        paired = []
        for i in range(self.ns):
            m = best_match_src[i]
            if best_match_trg[m] == i:
                paired.append((i,m))

        paired_toks = []
        if self.src_words and self.trg_words:
            paired_toks = [(self.src_words[i],self.trg_words[j]) for (i,j) in paired]
        else:
            paired_toks = paired
        return paired_toks


    def score_profile(self, w, topk = 10, domain = 'src'):
        if domain == 'src':
            w_idx      = self.src_word2ind[w]
            score_vec  = self.scores[w_idx,:]
            all_labels = self.trg_words
        else:
            w_idx     = self.trg_word2ind[w]
            score_vec = self.scores[:,w_idx]
            all_labels = self.src_words


        knn = np.argsort(score_vec)[::-1][:topk]
        vals = [score_vec[k] for k in knn]

        labels = [all_labels[k] for k in knn]
        fig, ax = plt.subplots()
        ax.bar(list(range(topk)), vals)
        ax.set_xticks(range(topk))
        ax.set_xticklabels(labels, rotation = 90)
        # show horizontal line with marginal prob.
        #if self.adjust == None:

        plt.show()

    def normalize_embeddings(self, xs = None, xt = None):
        if self.normalize_vecs:
            print("Normalizing embeddings with: {}".format(self.normalize_vecs))
        if self.normalize_vecs == 'whiten':
            print('Warning: whiten not yet implemented for OOV data')
            self._center_embeddings()
            self._whiten_embeddings()
        elif self.normalize_vecs == 'mean':
            self._center_embeddings()
        elif self.normalize_vecs == 'both':
            self._center_embeddings()
            self._scale_embeddings()
            self.solver.normalized = True
        elif self.normalize_vecs == 'whiten_zca':
            print('Warning: whiten zca not yet implemented for OOV data')
            self._center_embeddings()
            self._whiten_embeddings_zca()
        else:
            print('Warning: no normalization')

    def _center_embeddings(self):
        self.xs -= self.xs.mean(axis=0)
        self.xt -= self.xt.mean(axis=0)
        if self.xs_oov is not None:
            self.xs_oov -= self.xs_oov.mean(axis=0)
            self.xt_oov -= self.xt_oov.mean(axis=0)
        self.centered = True

    def _scale_embeddings(self):
        self.xs /= np.linalg.norm(self.xs, axis=1)[:,None]
        self.xt /= np.linalg.norm(self.xt, axis=1)[:,None]
        if self.xs_oov is not None:
            self.xs_oov /= np.linalg.norm(self.xs_oov, axis=1)[:,None]
            self.xt_oov /= np.linalg.norm(self.xt_oov, axis=1)[:,None]

    def _whiten_embeddings(self):
        """
            PCA whitening. https://stats.stackexchange.com/questions/95806/how-to-whiten-the-data-using-principal-component-analysis
            Uses PCA of covariance matrix Sigma = XX', Sigma = ULU'.
            Whitening matrix given by:
                W = L^(-1/2)U'

        """
        if not self.centered:
            raise ValueError("Whitenning needs centering to be done in advance")
        #self._center_vectors()
        n,d = self.xs.shape

        Cov_s = np.cov(self.xs.T)
        _, S_s, V_s = np.linalg.svd(Cov_s)
        W_s = (V_s.T/np.sqrt(S_s)).T
        assert np.allclose(W_s@Cov_s@W_s.T, np.eye(d)) # W*Sigma*W' = I_d

        Cov_t = np.cov(self.xt.T)
        _, S_t, V_t = np.linalg.svd(Cov_t)
        W_t = (V_t.T/np.sqrt(S_t)).T
        assert np.allclose(W_t@Cov_t@W_t.T, np.eye(d))

        self.xs = self.xs@W_s.T
        self.xt = self.xt@W_t.T

        assert np.allclose(np.cov(self.xs.T), np.eye(d))  # Cov(hat(x)) = I_d
        assert np.allclose(np.cov(self.xt.T), np.eye(d))

    def _whiten_embeddings_zca(self, lambd = 1e-8):
        """ ZCA whitening
            (C_xx+gamma I)^{-0.5}X
            (C_yy+gamma I)^{-0.5}Y
        """
        print('ZCA-Whitening')
        Cov_s = np.cov(self.xs.T)
        Cov_t = np.cov(self.xt.T)
        d = Cov_s.shape[0]

        W_s =  scipy.linalg.sqrtm(Cov_s + lambd*np.eye(d))
        W_t =  scipy.linalg.sqrtm(Cov_t + lambd*np.eye(d))

        self.xs = self.xs@W_s#.T
        self.xt = self.xt@W_t#.T


class procot_bilind(bilingual_mapping):
    """
        Bilingual lexical induction with Procrustes Optimal Transport.
    """
    def __init__(self, *args, **kwargs):
        super(procot_bilind, self).__init__(*args, **kwargs)

    def get_mapping(self, *args, **kwargs):
        return self.mapping

    def init_optimizer(self, *args, **kwargs):
        print('Initializing bilingual mapping with Procrustes OT')
        self.solver = invarot.optim.invariance_ot_solver(**kwargs)
        self.solver.accuracy_function = self._test_accuracy

    def fit(self, maxiter = 100, plot_every = 100, print_every = 10,
            verbose = True, *args, **kwargs):

        #test_fun = functools.partial(self.test_accuracy(G, ))
        if not self.solver:
            raise ValueError('Optimizer has not been initalized yet. Call init_optimizer before hand.')

        # 0. Pre-processing
        self.normalize_embeddings()

        # 1. Solve OT
        print('Solving optimization problem...')
        if hasattr(self,"P_init"):  #is not None:
            P_init = self.P_init
        else:
            P_init = None
        G,P = self.solver.solve(self.xs,self.xt, P_init = P_init, maxiter = maxiter,
                                plot_every = plot_every, print_every = print_every,
                                verbose = verbose)
        self.coupling   = G
        self.mapping    = P

        # 2. From Couplings to Translation Score
        print('Computing translation scores...')
        self.compute_scores(self.score_type, adjust = self.adjust)




class gromov_bilind(bilingual_mapping):
    """
        Bilingual lexical induction with Gromov Wasserstein.
    """
    def __init__(self, *args, **kwargs):
        super(gromov_bilind, self).__init__(*args, **kwargs)

        # Gromov Specific args
        # Maybe put entropic, entreg here?

    def init_optimizer(self, *args, **kwargs):
        print('Initializing Gromov-Wasserstein optimizer')
        self.solver = gromov_wass_solver(**kwargs)
        if self.test_dict is not None:
            self.solver.accuracy_function = self._test_accuracy
        else:
            self.solver.compute_accuracy = False

    def get_mapping(self, type = 'orthogonal', anchor_method = 'mutual_nn', max_anchors = None):
        """
            Infer mapping given optimal coupling
        """
        # Method 1: Orthogonal projection that best macthes NN
        self.compute_scores(score_type='coupling') # TO refresh
        if anchor_method == 'mutual_nn':
            pseudo = self.find_mutual_nn()#[:100]
        elif anchor_method == 'all':
            translations, oov = self.generate_translations()
            pseudo = [(k,v[0]) for k,v in translations.items()]
        if max_anchors:
            pseudo = pseudo[:max_anchors]
        print('Finding orthogonal mapping with {} anchor points via {}'.format(len(pseudo), anchor_method))
        if anchor_method in ['mutual_nn', 'all']:
            idx_src = [self.src_word2ind[ws] for ws,_ in pseudo]
            idx_trg = [self.trg_word2ind[wt] for _,wt in pseudo]
            xs_nn = self.xs[idx_src]
            xt_nn = self.xt[idx_trg]
            P = orth_procrustes(xs_nn, xt_nn)
        elif anchor_method == 'barycenter':
            ot_emd = ot.da.EMDTransport()
            ot_emd.xs_ = self.xs
            ot_emd.xt_ = self.xt
            ot_emd.coupling_= self.coupling
            xt_hat = ot_emd.inverse_transform(Xt=self.xt) # Maps target to source space
            P = orth_procrustes(xt_hat, self.xt)
        return P





    def fit(self, maxiter = 300, tol = 1e-9, print_every = None,
            plot_every = None, verbose = False, save_plots = None):
        """
            Wrapper function that computes all necessary steps to estimate
            bilingual mapping using Gromov Wasserstein.
        """
        print('Fitting bilingual mapping with Gromov Wasserstein')

        # 0. Pre-processing
        self.normalize_embeddings()
        #self.compute_monolingual_distances() # Done in optim

        # 1. Solve Gromov Wasserstein problem
        print('Solving optimization problem...')
        G = self.solver.solve(self.xs,self.xt,self.p,self.q,
                                maxiter = maxiter,
                                plot_every = plot_every, print_every = print_every,
                                verbose = verbose, save_plots = save_plots)
        self.coupling = G
        #
        # G = self.solve_gromovwass(entropic = entropic, reg = entreg,  verbose = verbose)
        # self.coupling = G

        # 3. From Couplings to Translation Score
        print('Computing translation scores...')
        print(self.score_type, self.adjust)
        self.compute_scores(self.score_type, adjust = self.adjust)

        self.mapping = self.get_mapping()


    def solve_gromovwass(self, entropic = True, reg = 1e-3, loss = 'square_loss',
                    max_iter = 1000, tol = 1e-9, verbose = False):
        """ DEPRECATED: NOW DONE IN-HOUSE IN OPTIM / OPTIM_GPU """

        # 3. Solve GW problem
        print(verbose)
        print('Solving GW {} problem ....'.format('regularized' if entropic else 'non-regularized'))
        if not entropic:
            G,log = ot.gromov.gromov_wasserstein(self.Cs, self.Ct, self.p, self.q, loss,
                                            log = True, verbose = verbose)
        else:
            G,log = ot.gromov.entropic_gromov_wasserstein(self.Cs, self.Ct, g, self.q, loss,
                                            max_iter = max_iter,
                                            tol = tol,
                                            epsilon=reg,
                                            log = True, verbose = verbose)


        #cost = np.sum(G * ot.gromov.tensor_square_loss(self.Cs, self.Ct, G))
        cost = log['gw_dist']

        # 4. Analyze solution
        if verbose:
            print('cost(G): {:8.2f}'.format(cost))
            print('G is p-feasible: {}'.format('Yes' if np.allclose(G.sum(1),self.p) else 'No'))
            print('G is q-feasible: {}'.format('Yes' if np.allclose(G.sum(0),self.q) else 'No' + ' ||sum G - q || = {:8.2e}'.format(np.linalg.norm(G.sum(0)-self.q))))
            plt.figure()
            plt.imshow(G, cmap='jet')
            plt.colorbar()
            plt.show()

        #self.coupling = G
        return G

    def local_matching(self, src_w, trg_w, k = 10, reg = 1e-3, max_iter = 100,
                            entropic = False, tol = 1e-8, verbose = False):
        """ Perform local matching: neihbors of src w to neighbors of trg w (incusive of src/trg)

                - k (int): number of nearest neighbors to match
        """

        # 1. Get neighbors
        src_idx = self.src_word2ind[src_w]
        trg_idx = self.trg_word2ind[trg_w]
        knn_src_idx = sorted(np.argpartition(self.Cs[src_idx,:], k, axis=0)[:k])
        knn_trg_idx = sorted(np.argpartition(self.Ct[trg_idx,:], k, axis=0)[:k])
        knn_src = [self.src_words[i] for i in knn_src_idx]
        knn_trg = [self.trg_words[i] for i in knn_trg_idx]
        print(knn_src)
        print(knn_trg)

        # 2. Compute local marginal distributions + Costs FIXME: Should we take subset of p/q and renormalize instead?
        p_loc = ot.unif(k)
        q_loc = ot.unif(k)
        Cs_loc = self.Cs[knn_src_idx,:][:,knn_src_idx]
        Ct_loc = self.Ct[knn_trg_idx,:][:,knn_trg_idx]
        print(Cs_loc.shape)
        print(Ct_loc.shape)

        # 3. Solve GW problem
        print('Solving GW problem....')
        if not entropic:
            G_local = ot.gromov.gromov_wasserstein(Cs_loc, Ct_loc, p_loc, q_loc,
                                        'square_loss',
                                        verbose = verbose)
        else:
            G_local = ot.gromov.entropic_gromov_wasserstein(Cs_loc, Ct_loc, p_loc, q_loc,
                                        'square_loss',
                                        max_iter = max_iter,
                                        tol = tol,
                                        epsilon = reg,
                                        verbose = verbose)


        # 4. Display
        fig, ax = plt.subplots(1,3, figsize=(15,5))
        ax[0].imshow(Cs_loc, cmap='jet')
        ax[0].set_title('NN of src word: {}'.format(src_w))
        ax[0].set_xticks(range(len(knn_src)))
        ax[0].set_xticklabels(knn_src,  rotation=90)
        ax[0].set_yticks(range(len(knn_src)))
        ax[0].set_yticklabels(knn_src)

        ax[1].imshow(Ct_loc, cmap='jet')
        ax[1].set_title('NN of trg word: {}'.format(trg_w))
        ax[1].set_xticks(range(len(knn_trg)))
        ax[1].set_xticklabels(knn_trg,  rotation=90)
        ax[1].set_yticks(range(len(knn_trg)))
        ax[1].set_yticklabels(knn_trg)

        ax[2].imshow(G_local, cmap='jet')
        ax[2].set_title('Optimal Local G')
        ax[2].set_xticks(range(len(knn_trg)))
        ax[2].set_xticklabels(knn_trg,  rotation=90)
        ax[2].set_yticks(range(len(knn_src)))
        ax[2].set_yticklabels(knn_src)
        #ax.colorbar()
        plt.show()

        return G_local, knn_src_idx, knn_trg_idx

    def compute_hierarchical_gw(self, global_entropic = True, global_reg = 1e-3,
                                      local_entropic = False, local_reg = 1e-3,
                                      local_knn = 10,
                                      loss = 'square_loss',
                                      max_iter = 1000, tol = 1e-9, verbose = False):


        # 1. Solve Initial GW problem
        if self.gw_coupling:
            print('Global GW found, WONT recompute')
        else:
            self.solve_gromovwass(entropic = global_entropic, eps = global_reg, max_iter = max_iter, verbose = verbose)

        # 2. Computes scores
        self.compute_scores('coupling', adjust = 'csls', verbose = verbose)

        # 3. Find anchors
        pairs = self.find_mutual_nn()
        print('Mutual NN pairs:', pairs)

        # 4. Find local matchings for anchors
        hier_G = np.zeros_like(self.gw_coupling)
        for i, (w_src, w_trg) in enumerate(pairs):
            g_loc, src_idx, trg_idx = self.local_matching(w_src, w_trg, k=local_knn,
                                      entropic = local_entropic, reg = local_reg,
                                      verbose = verbose)
            hier_G[np.ix_(src_idx,trg_idx)] = g_loc
            if i > 20:
                break


        return hier_G




#====================


def GW_word_mapping(xs, xt, src_words, tgt_words,
                    eps = 1e-3, metric = 'cosine', loss = 'square_loss',
                    max_iter = 1000, tol = 1e-9, verbose = False):
    """
        All-in-one wrapper function to compute GW for word embedding mapping.
        Computes distance matrices, marginal distributions, solves GW problem and returns
        translations.
    """


    # 1. Compute distance matrices
    print('Computing distance matrices....', end = '')
    C1 = sp.spatial.distance.cdist(xs, xs, metric = metric)
    C2 = sp.spatial.distance.cdist(xt, xt, metric = metric)
    print('Done!')

    # 2. Compute marginal distributions
    p = ot.unif(xs.shape[0])
    q = ot.unif(xt.shape[0])

    # 3. Solve GW problem
    print('Solving GW problem....')
    G = ot.gromov_wasserstein(C1, C2, p, q, loss,
                                    max_iter = max_iter,
                                    tol = tol,
                                    epsilon=eps,
                                    verbose = verbose)

    cost = np.sum(G * tensor_square_loss(C1, C2, G))

    # 4. Analyze solution
    print('cost(G): {:8.2f}'.format(cost))
    print('G is p-feasible: {}'.format('Yes' if np.allclose(G.sum(1),p) else 'No'))
    print('G is q-feasible: {}'.format('Yes' if np.allclose(G.sum(0),q) else 'No'))
    plt.figure()
    plt.imshow(G, cmap='jet')
    plt.colorbar()
    plt.show()

    # 5. Compute word translations from mapping
    mapping = translations_from_coupling(G, src_words, tgt_words, verbose = True)

    return mapping, G, cost

def translations_from_coupling(G, src_words = None, tgt_words=None, verbose = False):
    """
        Returns pairs of matched (row, col) pairs according to some criterion
    """
    # Naive method: look for unambiguous words who are mutually NN
    G.max(0)
    n_s, n_t = G.shape
    best_match_src = G.argmax(1) # Best match for each source word
    best_match_tgt = G.argmax(0)

    paired = []
    for i in range(n_s):
        m = best_match_src[i]
        if verbose:
            k = 10
            topk_idx = np.argpartition(G[i,:], -k)[-k:]
            topk_idx_sort = topk_idx[np.argsort(-G[i,topk_idx])] # With - to get descending order
            print('{:20s} -> {}'.format(src_words[i],','.join([tgt_words[m] for m in topk_idx_sort])))
        if best_match_tgt[m] == i:
            paired.append((i,m))

    paired_toks = []
    if src_words and tgt_words:
        paired_toks = [(src_words[i],tgt_words[j]) for (i,j) in paired]
    else:
        paired_toks = paired
    return paired_toks

def compute_precision(gold_dict, src_word2ind, trg_word2ind, src_words, trg_words, scores, args, BATCH_SIZE=20, verbose = False):
    """
        gold_dict is given as dict of *indices* of words.

        scores are similarities (not distances!), so pairs are made based on
        translation with highest score

    """
    # TODO: Generalize to compute Prec @5, @10.
    # Read dictionary and compute coverage
#     dictf = open(args.dictionary, encoding=args.encoding, errors='surrogateescape')
#     src2trg = collections.defaultdict(set)
#     oov = set()
#     vocab = set()
#     for line in dictf:
#         src, trg = line.split()
#         try:
#             src_ind = src_word2ind[src]
#             trg_ind = trg_word2ind[trg]
#             src2trg[src_ind].add(trg_ind)
#             vocab.add(src)
#         except KeyError:
#             oov.add(src)
#     oov -= vocab  # If one of the translation options is in the vocabulary, then the entry is not an oov
#     coverage = len(src2trg) / (len(src2trg) + len(oov))

    oov = set()
    correct = 0
    n,m = scores.shape # The actual size of mapping computed, might be smaller that total size of dict

    precisions = {}

    if verbose:
        print('@{:2} {:10} {:30} {:30}'.format('k', 'Src','Predicted','Gold'))
        print_row = '{:2} {:10} {:30} {:30} {}'

    for k in [1,5,10]:
        correct = 0
        for src_idx,tgt_idx in gold_dict.items():
            if src_idx > n or np.all([e>m for e in tgt_idx]):#Src word not in mapping
                oov.add(src_idx)
                continue
            else:
                #print(k, src_idx, tgt_idx)
                knn = np.argpartition(scores[src_idx,:], -k)[-k:] # argpartition returns top k not in order
                knn_sort = knn[np.argsort(-scores[src_idx,knn])] # With - to get descending order

                #print(src_words[src_idx], knn_sort, ' '.join([trg_words[v] for v in tgt_idx]))
                #break
                if set(knn_sort).intersection(tgt_idx):
                    correct +=1
                    correct_string = ' '
                else:
                    correct_string = 'X'
                if verbose:
                    src_str = src_words[src_idx]
                    pred_str =  ','.join([trg_words[k] for k in knn_sort])
                    gold_str = ','.join([trg_words[k] for k in tgt_idx])
                    print(print_row.format(k,src_str,pred_str,gold_str,correct_string))


        coverage = len(gold_dict.keys()) / (len(gold_dict.keys()) + len(oov))
        if verbose:
            print('Coverage: {:7.2%}  Precision @{:2}:{:7.2%}'.format(coverage, k, correct / len(gold_dict)))
        precisions[k] = correct / len(gold_dict)
    return  precisions




def test_csls():
    toy_scores = np.array([
        [1, 1, 1, 1],
        [1, 0, 0, 0],
        [0, 2, 2, 0]
    ])

    csls(toy_scores, 2)

def csls(scores, knn = 5):
        """
            Adapted from Conneau et al.

            rt = [1/k *  sum_{zt in knn(xs)} score(xs, zt)
            rs = [1/k *  sum_{zs in knn(xt)} score(zs, xt)
            csls(x_s, x_t) = 2*score(xs, xt) - rt - rs

        """
        def mean_similarity(scores, knn, axis = 1):
            nghbs = np.argpartition(scores, -knn, axis = axis) # for rows #[-k:] # argpartition returns top k not in order but it's efficient (doesnt sort all rows)
            # TODO: There must be a faster way to do this slicing
            if axis == 1:
                nghbs = nghbs[:,-knn:]
                #print(nghbs.shape)
                nghbs_score = np.concatenate([row[indices] for row, indices in zip(scores, nghbs)]).reshape(nghbs.shape)
            else:
                nghbs = nghbs[-knn:,:].T
                #print(nghbs.shape)
                nghbs_score = np.concatenate([col[indices] for col, indices in zip(scores.T, nghbs)]).reshape(nghbs.shape)

            return nghbs_score.mean(axis = 1)
        # 1. Compute mean similarity return_scores
        src_ms = mean_similarity(scores, knn, axis = 1)
        trg_ms = mean_similarity(scores, knn, axis = 0)
        # 2. Compute updated scores
        normalized_scores = ((2*scores - trg_ms).T - src_ms).T
        return normalized_scores

def csls_sparse(X, Y, idx_x, idx_y, knn = 10):
    def mean_similarity_sparse(X, Y, seeds, knn, axis = 1, metric = 'cosine'):
        if axis == 1:
            dists = sp.spatial.distance.cdist(X[seeds,:], Y, metric=metric)
        else:
            dists = sp.spatial.distance.cdist(X, Y[seeds,:], metric=metric).T
        nghbs = np.argpartition(dists, knn, axis = 1) # for rows #[-k:] # argpartition returns top k not in order but it's efficient (doesnt sort all rows)
        nghbs = nghbs[:,:knn]
        nghbs_dists = np.concatenate([row[indices] for row, indices in zip(dists, nghbs)]).reshape(nghbs.shape)
        nghbs_sims  = 1 - nghbs_dists
        return nghbs_sims.mean(axis = 1)

    src_ms = mean_similarity_sparse(X, Y, idx_x, knn,  axis = 1)
    trg_ms = mean_similarity_sparse(X, Y, idx_y, knn,  axis = 0)
    sims =  1 - sp.spatial.distance.cdist(X[idx_x,:], Y[idx_y,:])
    normalized_sims = ((2*sims - trg_ms).T - src_ms).T
    print(normalized_sims)
    nn = normalized_sims.argmax(axis=1).tolist()
    return nn

def compute_accuracy(gold_dict, src_word2ind, trg_word2ind, src_words, trg_words, scores, args, k=5, BATCH_SIZE=20):
    """
        gold_dict is given as dict of *indices* of words.

    """
    # TODO: Generalize to compute Prec @5, @10.
    # Read dictionary and compute coverage
#     dictf = open(args.dictionary, encoding=args.encoding, errors='surrogateescape')
#     src2trg = collections.defaultdict(set)
#     oov = set()
#     vocab = set()
#     for line in dictf:
#         src, trg = line.split()
#         try:
#             src_ind = src_word2ind[src]
#             trg_ind = trg_word2ind[trg]
#             src2trg[src_ind].add(trg_ind)
#             vocab.add(src)
#         except KeyError:
#             oov.add(src)
#     oov -= vocab  # If one of the translation options is in the vocabulary, then the entry is not an oov
#     coverage = len(src2trg) / (len(src2trg) + len(oov))

    oov = set()
    correct = 0
    n,m = scores.shape # The actual size of mapping computed, might be smaller that total size of dict
    for src_idx,tgt_idx in gold_dict.items():
            #srx_idx = src_word2ind[src_w]
        #tgt_idxs = set([trg_word2ind[w] for w in tgt_w])
        # if src_idx >
        np.any([e<m for e in tgt_idx])

        if src_idx > n or np.all([e>m for e in tgt_idx]):#Src word not in mapping
            oov.add(src_idx)
            continue
        else:
            print(k, src_idx, tgt_idx)
            knn = np.argpartition(scores[src_idx,:], -k)[-k:]
            knn_sort = knn[np.argsort(-scores[src_idx,knn])] # With - to get descending order
            #print(src_words[src_idx], knn_sort, ' '.join([trg_words[v] for v in tgt_idx]))
            #break
            if set(knn_sort).intersection(tgt_idx):
                correct +=1

    coverage = len(gold_dict.keys()) / (len(gold_dict.keys()) + len(oov))
    if verbose:
        print('Coverage: {0:7.2%}  Accuracy:{1:7.2%}'.format(coverage, correct / len(gold_dict)))
    return  correct / len(gold_dict)
