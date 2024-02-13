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
                 'tie', 'motorcycle', 'bicycle', 'glove', 'watch', 'belt', 'leg warmer', 'tights, stockings', 'sock', 'shoe', 'bag, wallet', 'scarf', 'umbrella'),
        palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                 [107, 142, 35], [152, 251, 152], [70, 130, 180],
                 [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                 [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]
                 [64, 128, 0], [192, 192, 32], [128, 96, 192], [64, 0, 128],
                 [64, 64, 32], [0, 224, 192], [192, 0, 0], [192, 64, 160]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='_seg.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)