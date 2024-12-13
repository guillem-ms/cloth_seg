from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class PogBgDataset(BaseSegDataset):
    """POG Background dataset.

    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is
    fixed to '.png' for POG dataset.
    """

    METAINFO = dict(
        classes=("background", "garment"), palette=[[0, 0, 0], [250, 50, 83]]
    )

    def __init__(self, img_suffix=".jpg", seg_map_suffix=".png", **kwargs) -> None:
        super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
