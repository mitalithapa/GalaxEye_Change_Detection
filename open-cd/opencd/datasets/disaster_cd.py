
from opencd.registry import DATASETS
from opencd.datasets.basecddataset import _BaseCDDataset

@DATASETS.register_module(force=True)
class DisasterCDDataset(_BaseCDDataset):
    METAINFO = dict(classes=('unchanged', 'changed'), palette=[[0, 0, 0], [255, 255, 255]])
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
