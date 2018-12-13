"""

    Main script for inferring correspondences across domains by using the
    Gromov-Wasserstein distanceself.

    Parts of the machinery to load / evaluate word embeddings where built upon
    the very thorough codebase by Artetxe https://github.com/artetxem

"""
import sys
import os
import argparse
import collections
from collections import defaultdict
from time import time
import pickle

import scipy as sp
import numpy as np
import matplotlib
import matplotlib.pylab as plt

import pdb
import argparse
import ot

from src.bilind import gromov_bilind, bilingual_mapping
import src.embeddings as embeddings


def dump_results(outdir, args, optim_args, acc, BLI):
    results = {'acc': acc, 'args': vars(args), 'optim_args':  vars(optim_args)}#, 'G': BLI.coupling}
    if BLI.mapping is not None:
        results['P'] = BLI.mapping
    np.save(os.path.join(outdir, "coupling"), BLI.coupling)
    dump_file = os.path.join(outdir, "results.pkl")
    pickle.dump(results, open(dump_file, "wb"))

def load_results(outdir, BLI):
    dump_file = os.path.join(outdir, "results.pkl")
    results = pickle.load(open(dump_file, "rb"))
    BLI.mapping  = results['P']
    BLI.coupling = results['G']
    return BLI

def load_vectors(args):
    """
        Assumes file structure as in MUSE repo.
    """
    dict_fold = 'train' # which fold of the data will be used to produce results
    if args.task == 'conneau':
        data_dir = os.path.join(args.data_dir, 'MUSE')
        dict_dir = os.path.join(data_dir, 'crosslingual/dictionaries/')
        src_path = os.path.join(data_dir, 'wiki.' + args.src_lang + '.vec')
        trg_path = os.path.join(data_dir, 'wiki.' + args.trg_lang + '.vec')
        src_freq_path = None
        trg_freq_path = None
        if dict_fold == 'test':
            postfix = '.5000-6500.txt'
        elif dict_fold == 'train':
            postfix = '.0-5000.txt'
        else:
            raise ValueError('Unrecognized dictionary fold for evaluation')
    elif args.task == 'dinu':
        data_dir = os.path.join(args.home_dir,'pkg/vecmap/data')
        dict_dir = os.path.join(data_dir, 'dictionaries/')
        src_path = os.path.join(data_dir, 'embeddings/original', args.src_lang + '.emb.txt')
        trg_path = os.path.join(data_dir, 'embeddings/original', args.trg_lang + '.emb.txt')
        src_freq_path = None
        trg_freq_path = None
        postfix  = '.{}.txt'.format(dict_fold)
    elif args.task == 'zhang':
        order = [args.src_lang,args.trg_lang]
        if args.src_lang == 'en':
            order = order[::-1]
        data_dir = os.path.join(args.home_dir,'pkg/UBiLexAT/data/','-'.join(order))
        dict_dir = data_dir
        src_path = os.path.join(data_dir, 'word2vec.' + args.src_lang)
        trg_path = os.path.join(data_dir, 'word2vec.' + args.trg_lang)
        src_freq_path = os.path.join(data_dir, 'vocab-freq.' + args.src_lang)
        trg_freq_path = os.path.join(data_dir, 'vocab-freq.' + args.trg_lang)
        postfix  = '.train.txt'

    srcfile = open(src_path, encoding=args.encoding, errors='surrogateescape')
    trgfile = open(trg_path, encoding=args.encoding, errors='surrogateescape')
    src_words, xs = embeddings.read(srcfile, args.maxs)
    trg_words, xt = embeddings.read(trgfile, args.maxt)

    if src_freq_path:
        with open(src_freq_path, encoding=args.encoding, errors='surrogateescape') as f:
            lines = [a.split(' ') for a in f.read().strip().split('\n')]
            freq_src = {k: int(v) for (k,v) in lines}

        with open(trg_freq_path, encoding=args.encoding, errors='surrogateescape') as f:
            lines = [a.split(' ') for a in f.read().strip().split('\n')]
            freq_trg = {k: int(v) for (k,v) in lines}

    # Build word to index map
    src_word2ind = {word: i for i, word in enumerate(src_words)}
    trg_word2ind = {word: i for i, word in enumerate(trg_words)}

    if args.task == 'zhang':
        dict_path = os.path.join(dict_dir,  'all.' + '-'.join(order) + '.lex')
        flip = False
    elif args.task == 'dinu' and args.src_lang != 'en':
        # Only has dicts in one direction, flip
        dict_path = os.path.join(dict_dir, args.trg_lang + '-' + args.src_lang + postfix)
        flip = True
    else:
        dict_path = os.path.join(dict_dir, args.src_lang + '-' + args.trg_lang + postfix)
        flip = False

    if not os.path.exists(dict_path):
        print('Warning: no dict found, will continue in unsupervised mode')
        return xs, xt, src_words, trg_words,  src_word2ind, trg_word2ind, None

    dictf = open(dict_path, encoding=args.encoding, errors='surrogateescape')
    src2trg = collections.defaultdict(set)
    oov = set()
    vocab = set()
    max_srcind = 0 # These are mostly for debug
    max_trgind = 0
    for line in dictf:
        splitted = line.split()
        if len(splitted) > 2:
            # Only using first translation if many are provided
            src, trg = splitted[:2]
        elif len(splitted) == 2:
            src, trg = splitted
        else:
            # No translation? Only happens for Zhang data so far
            continue
        if flip: src, trg = trg, src
        try:
            src_ind = src_word2ind[src]
            trg_ind = trg_word2ind[trg]
            src2trg[src_ind].add(trg_ind)
            vocab.add(src)
            max_srcind = max(max_srcind, src_ind)
            max_trgind = max(max_trgind, trg_ind)
        except KeyError:
            oov.add(src)
    oov -= vocab  # If one of the translation options is in the vocabulary, then the entry is not an oov
    coverage = len(src2trg) / (len(src2trg) + len(oov))

    print('Max test dict src/trg indices: {}/{}'.format(max_srcind, max_trgind))

    print(
        'Coverage (pairs from test dict contained in src/trg emb): {:8.2f}%'.format(100 * coverage))
    if coverage < .001:
        raise ValueError(
            'Coverage of task vocabulary is too low. Increase maxs and maxt!')
    return xs, xt, src_words, trg_words,  src_word2ind, trg_word2ind, src2trg

def parse_args():
    parser = argparse.ArgumentParser(description='Word embedding alignment with GW',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    ### General Task Options
    general = parser.add_argument_group('General task options')
    general.add_argument('--debug',action='store_true',
                    help='trigger debugging mode (saving to /tmp/)')
    general.add_argument('--src_lang', type=str,
                         default='en', help='source language')
    general.add_argument('--trg_lang', type=str,
                         default='es', help='target language')
    general.add_argument('--data_dir', type=str, default='data/raw',
                         help='where word embedding data is located (i.e. path to MUSE/data dir)')
    general.add_argument('--load', action='store_true',
                         help='load previously trained model')
    general.add_argument('--task', type=str, default='conneau', choices = ['dinu','zhang','conneau'],
                         help='which task to test on')
    general.add_argument('--encoding', type=str,
                         default='utf-8', help='embedding encoding')
    general.add_argument('--maxs', type=int, default=2000,
                         help='use only first k embeddings from source [default: 2000]')
    general.add_argument('--maxt', type=int, default=2000,
                         help='use only first k embeddings from target [default: 2000]')
    general.add_argument('--distribs', type=str, default='uniform',
                         help='p/q distributions to use [default: uniform]')
    general.add_argument('--normalize_vecs', type=str, default='both',
                         choices=['mean','both','whiten','whiten_zca','none'], help='whether to normalize embeddings')
    general.add_argument('--score_type', type=str, default='coupling', choices=[
                       'coupling','transported','distance'], help='what variable to use as the basis for translation scores')
    general.add_argument('--adjust', type=str, default='none', choices=[
                       'csls','isf','none'], help='What type of neighborhood adjustment to use')
    general.add_argument('--maxiter', type=int, default=1000, help='Max number of iterations for optimization problem')


    #### PATHS
    general.add_argument('--chkpt_path', type=str,
                         default='checkpoints', help='where to save the snapshot')
    general.add_argument('--results_path', type=str, default='out',
                         help='where to dump model config and epoch stats')
    general.add_argument('--log_path', type=str, default='log',
                         help='where to dump training logs  epoch stats (and config??)')
    general.add_argument('--summary_path', type=str, default='results/summary.csv',
                         help='where to dump model config and epoch stats')

    ### SAVING AND CHECKPOINTING
    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency during train (in iters)')
    parser.add_argument('--save_freq', type=int, default=100,
                        help='checkpoint save frequency during train (in  iters)')
    parser.add_argument('--plot_freq', type=int, default=100,
                        help='plot frequency during train (in  iters)')

    #############   Gromov-specific Optimization Args ###############
    gromov_optim = parser.add_argument_group('Gromov Wasserstein Optimization options')

    gromov_optim.add_argument('--metric', type=str, default='cosine', choices=[
                           'euclidean', 'sqeuclidean', 'cosine'], help='metric to use for computing vector distances')
    gromov_optim.add_argument('--normalize_dists', type=str, default='mean', choices = ['mean','max','median','none'],
                       help='method to normalize distance matrices')
    gromov_optim.add_argument('--no_entropy', action='store_false', default=True, dest='entropic',
                       help='do not use entropic regularized Gromov-Wasserstein')
    gromov_optim.add_argument('--entreg', type=float, default=5e-4,
                       help='entopy regularization for sinkhorn')
    gromov_optim.add_argument('--tol', type=float, default=1e-8,
                       help='stop criterion tolerance for sinkhorn')
    gromov_optim.add_argument('--gpu', action='store_true',
                       help='use CUDA/GPU for sinkhorn computation')

    args = parser.parse_args()

    if args.debug:
        args.verbose      = True
        args.chkpt_path   = '/tmp/'
        args.results_path = '/tmp/'
        args.log_path     = '/tmp/'
        args.summary_path = '/tmp/'

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))


    optimp = gromov_optim
    optimp.normalize_vecs  = args.normalize_vecs
    optimp.normalize_dists = args.normalize_dists

    optim_args = argparse.Namespace(
        **{a.dest: getattr(args, a.dest, None) for a in optimp._group_actions})
    data_args = argparse.Namespace(
        **{a.dest: getattr(args, a.dest, None) for a in general._group_actions})

    return args, optim_args

def make_path(root, args):
    if root is None:
        return None
    topdir = '_'.join([args.src_lang, args.trg_lang, str(args.maxs)])
    method = 'gromov'
    params = {  # Subset of parameters to put in filename
                'entreg': 'ereg',
                'tol': 'tol',
    }
    subdir = [method, args.normalize_vecs, args.metric, args.distribs]
    for arg, name in params.items():
        val = getattr(args, arg)
        subdir.append(params[arg] + '_' + str(val))
    subdir = '_'.join(subdir)
    path = os.path.join(root, args.task, topdir, subdir)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def print_header(method):
    print('='*80)
    print('='*13 +'  Bilingual Lexical Induction with Gromov-Wasserstein  ' +'='*12)
    print('='*80)

def main():
    """

        Pass outpath=checkpotins/bla to solve()
        Save progress plots, and current G and P in a pkl file:
        (it, G_t, P_t, lambda_G, lambda_P, ent_reg, ....)

        Add restarting from checkpoint:

        Saving to out:
            - history plot
            - final scores
            - tranlsations?
            - model? Popt Gopt
    """
    args, optim_args = parse_args()

    # Read Word Embeddings
    xs, xt, src_words, trg_words, src_word2ind, trg_word2ind, src2trg = \
        load_vectors(args)

    outdir   = make_path(args.results_path, args)
    chkptdir = make_path(args.chkpt_path, args)

    print('Saving checkpoints to: {}'.format(chkptdir))
    print('Saving results to: {}'.format(outdir))

    # Instantiate Bilingual Lexical Induciton Object
    BLI = gromov_bilind(xs, xt, args.src_lang, args.trg_lang, src_words, trg_words,
                        src_word2ind, trg_word2ind, src2trg,
                        metric = args.metric, normalize_vecs = args.normalize_vecs,
                        normalize_dists = args.normalize_dists,
                        score_type = args.score_type, adjust = args.adjust,
                        distribs = args.distribs)
    BLI.init_optimizer(**vars(optim_args)) # FIXME: This is ugly. Get rid of it

    if (not args.load) or (not os.path.exists(os.path.join(outdir, "results.pkl"))):
        if args.load:
            print('Could not load!!!')
        print('Will train from scratch')
        start = time()
        BLI.fit(maxiter=args.maxiter, plot_every=args.plot_freq,
                print_every=args.print_freq, verbose=True, save_plots = outdir)
        plt.close('all')
        print('Total elapsed time: {}s'.format(time() - start))
        if outdir:
            BLI.solver.plot_history(save_path=os.path.join(outdir, 'history.pdf'))
            acc = 0
            dump_results(outdir, args, optim_args, acc, BLI)
        else:
            BLI.solver.plot_history()
        print('Done!')
    else:
        print('Will load pre-solved solution from: ', outdir)
        BLI = load_results(outdir, BLI)


    ### Infer mapping from coupling - there's many ways to do this.
    #BLI.mapping = BLI.get_mapping(anchor_method = 'mutual_nn', max_anchors = 5000)
    #BLI.mapping = BLI.get_mapping(anchor_method = 'barycenter')
    #BLI.mapping = BLI.get_mapping(anchor_method = 'all')
    BLI.mapping = BLI.get_mapping(anchor_method = 'mutual_nn', max_anchors = None)

    acc_file = os.path.join(outdir, 'accuracies.tsv')
    acc_dict = {}
    print('Results on test dictionary for fitting vectors: (via coupling)')
    acc_dict['coupling'] = BLI.test_accuracy(verbose=True, score_type = 'coupling')
    print('Results on test dictionary for fitting vectors: (via coupling + csls)')
    acc_dict['coupling_csls'] = BLI.test_accuracy(verbose=True, score_type = 'coupling', adjust = 'csls')

    print('Results on test dictionary for fitting vectors: (via bary projection)')
    acc_dict['bary'] = BLI.test_accuracy(verbose=True, score_type = 'barycentric')
    print('Results on test dictionary for fitting vectors: (via bary projection + csls)')
    acc_dict['bary_csls'] = BLI.test_accuracy(verbose=True, score_type = 'barycentric', adjust = 'csls')

    print('Results on test dictionary for fitting vectors: (via orth projection)')
    acc_dict['proj'] = BLI.test_accuracy(verbose=True, score_type = 'projected')
    print('Results on test dictionary for fitting vectors: (via orth projection + csls)')
    acc_dict['proj_csls'] = BLI.test_accuracy(verbose=True, score_type = 'projected', adjust = 'csls')

    if outdir:
        print('Saving accuacy results')
        with open(acc_file, 'w') as f:
            for k,acc in acc_dict.items():
                f.write('\t'.join([k] + ['{:4.2f}'.format(100*v) for v in acc.values()]) + '\n')
        print('Saving in-vocabulary translations and mapped vectors')
        translation_file = os.path.join(outdir, "translations_transductive.tsv")
        BLI.dump_translations(src2trg, translation_file)
        #BLI.export_mapped(iov_mode='matched', outf=outdir, suffix = 'match') # Only needed for debug/analysis purposes
        BLI.export_mapped(iov_mode='projection', outf=outdir, suffix = 'proj')


    ### STEP 2: Out-of sample vectors
    print('************')
    print('Compute now for all vectors')
    # Read Word Embeddings
    argsc = argparse.Namespace(**vars(args))
    argsc.maxs = max(200000, args.maxs)
    argsc.maxt = max(200000, args.maxt)

    BLI.load_oov_data(*load_vectors(argsc), keep_original = True, normalize=True)
    #BLI.normalize_embeddings()

    print('Projecting and dumping....')
    if BLI.mapping is not None:
        BLI.export_mapped(iov_mode='projection', outf=outdir, suffix = 'proj-proj')
        #BLI.export_mapped(iov_mode='barycentric', outf=outdir, suffix = 'bary-proj')
        #BLI.export_mapped(iov_mode='matched', outf=outdir, suffix = 'match-proj')
        print('Done!')


if __name__ == "__main__":
    main()
