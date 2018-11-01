# -*- coding: utf-8 -*-
"""

"""

from .gw_optim import orth_procrustes

# from . import bregman
# from . import da
# from .bregman import sinkhorn
# from .da import sinkhorn_lpl1_mm
#
# from . import utils
# from .utils import dist, to_gpu, to_np





__all__ = ["utils", "dist", "sinkhorn",
           "sinkhorn_lpl1_mm", 'bregman', 'da', 'to_gpu', 'to_np']
