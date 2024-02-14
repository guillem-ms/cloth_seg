from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class FashionpediaDataset(BaseSegDataset):
    """Fashionpedia dataset.

    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is
    fixed to '_seg.png' for Fashionpedia dataset.
    """
    METAINFO = dict(
        classes=('shirt, bluse', 'top, t-shirt, sweatshirt', 'sweater', 'cardigan', 'jacket', 'vest',
                 'pants', 'shorts', 'skirt', 'coat', 'dress', 'jumpsuit', 'cape', 'glasses', 'hat', 'headband, head covering, hair accessory', 
                 'tie', 'glove', 'watch', 'belt', 'leg warmer', 'tights, stockings', 'sock', 'shoe', 'bag, wallet', 'scarf', 'umbrella'),
        palette=[[0, 0, 85], [0, 0, 170], [0, 0, 255], [0, 85, 0], [0, 85, 85], [0, 85, 170], [0, 85, 255], [0, 170, 0],
                 [0, 170, 85], [0, 170, 170], [0, 170, 255], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
                 [85, 0, 0], [85, 0, 85], [85, 0, 170], [85, 0, 255], [85, 85, 0], [85, 85, 85], [85, 85, 170], [85, 85, 255], 
                 [85, 170, 0], [85, 170, 85], [85, 170, 170], [85, 170, 255]]
    )

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='_seg.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)