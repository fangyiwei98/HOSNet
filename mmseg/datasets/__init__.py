# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS, PIPELINES, build_dataloader, build_dataset
from .cityscapes import CityscapesDataset
from .custom import CustomDataset
from .dataset_wrappers import (ConcatDataset, MultiImageMixDataset,
                               RepeatDataset)
from .isprs import ISPRSDataset
from .loveda import LoveDADataset


from .pv_forAdap import PVDataset_forAdap
from .LoveDA_forAdap import LoveDADataset_forAdap

from .PC_forAdap import PCDataset_forAdap
from .GR_forAdap import GRDataset_forAdap
from .SR_forAdap import SRDataset_forAdap
from .casid import CasidDataset
from .Casid_forAdap import CASIDDataset_forAdap



__all__ = [
    'CustomDataset', 'build_dataloader', 'ConcatDataset', 'RepeatDataset',
    'DATASETS', 'build_dataset', 'PIPELINES', 'CityscapesDataset',
    'LoveDADataset', 'MultiImageMixDataset',
    'ISPRSDataset', 'PVDataset_forAdap', 'LoveDA_forAdap',
    'PCDataset_forAdap', 'GRDataset_forAdap', 'SRDataset_forAdap',
    'CasidDataset', 'CASIDDataset_forAdap'
]

'''__all__ = [
    'CustomDataset', 'build_dataloader', 'ConcatDataset', 'RepeatDataset',
    'DATASETS', 'build_dataset', 'PIPELINES', 'CityscapesDataset',
    'PascalVOCDataset', 'ADE20KDataset', 'PascalContextDataset',
    'PascalContextDataset59', 'ChaseDB1Dataset', 'DRIVEDataset', 'HRFDataset',
    'STAREDataset', 'DarkZurichDataset', 'NightDrivingDataset',
    'COCOStuffDataset', 'LoveDADataset', 'MultiImageMixDataset',
    'ISPRSDataset', 'PVDataset_forAdap', 'LoveDA_forAdap', 
    'PCDataset_forAdap', 'GRDataset_forAdap', 'SRDataset_forAdap'
]'''
