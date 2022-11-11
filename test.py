import os, cv2
from tqdm import tqdm

from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# DefaultPredictor_GLCC
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T
import torch

# GLCC
from saba_dataset import get_saba_dicts
import rcnn_glcc



class DefaultPredictor_GLCC:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.

    Compared to using the model directly, this class does the following additions:

    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take one input image and produce a single output, instead of a batch.

    This is meant for simple demo purposes, so it does the above steps automatically.
    This is not meant for benchmarks or running complicated inference logic.
    If you'd like to do anything more complicated, please refer to its source code as
    examples to build and use the model manually.

    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.

    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            
            if self.model.glcc_on and self.model.glcc_output:
                predictions, pred_fish = self.model([inputs])  # List[ Dict('instances') ], pred_fishes:tensor
                return predictions, int(pred_fish)
            else:
                predictions = self.model([inputs])[0]


def _detection_evaluation( conf_name:str='./conf/rcnn.yaml', glcc_on:bool=False ):
    # Check-up
    if 'rcnn' in conf_name:
        assert glcc_on == False
    if 'glcc' in conf_name:
        assert glcc_on == True

    cfg = get_cfg()
    cfg.merge_from_file( conf_name )
    cfg.MODEL.GLCC_ON = glcc_on
    cfg.MODEL.GLCC_OUTPUT = False  # Detection output only
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    predictor = DefaultPredictor_GLCC(cfg)
    evaluator = COCOEvaluator( cfg.DATASETS.TEST[0], output_dir=cfg.OUTPUT_DIR)
    test_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0] )
    print( inference_on_dataset(predictor.model, test_loader, evaluator) )
    

def _accuracy_evaluation( conf_name:str='./conf/glcc.yaml' ):
    assert 'glcc' in conf_name
    
    cfg = get_cfg()
    cfg.merge_from_file( conf_name )
    cfg.MODEL.GLCC_ON = True
    cfg.MODEL.GLCC_OUTPUT = True  # Get GLCC output
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    predictor = DefaultPredictor_GLCC(cfg)
        
    # Show some results
    #dataset_dicts = get_saba_dicts("./data/test")  # 2017
    dataset_dicts = get_saba_dicts("./data/saba_20220930/test")  # 2022
    num_correct = 0


    for d in tqdm( dataset_dicts ):
        _fn = d["file_name"]
        im = cv2.imread( _fn )
        outputs, pred_fish = predictor(im)
        
        # Accuracy
        if d['fish_class']==pred_fish:
            num_correct += 1
                    
    # Result
    print('Accuracy: {}'.format( num_correct / len(dataset_dicts) ) )

    
    
if __name__ == '__main__':
    # Parameters
    #conf_names = ['./conf/rcnn_2017.yaml', './conf/glcc_2017.yaml']    
    conf_names = ['./conf/rcnn_2022.yaml', './conf/glcc_2022.yaml']

    # Register saba dataset to detectron2
    for year in ['2017', '2022']:
        if year=='2017':
            data_path = './data/'
        elif year=='2022':
            data_path = './data/saba_20220930/'
            
        data_tag = 'saba_{}_'.format(year)
        for d in ["train", "test"]:
            DatasetCatalog.register(data_tag + d, lambda d=d: get_saba_dicts(data_path + d) )
            MetadataCatalog.get(data_tag + d).set(thing_classes=['red','fish'])

    # Evaluations
    _detection_evaluation( conf_names[0], glcc_on=False )  # RCNN
    _detection_evaluation( conf_names[1], glcc_on=True )   # RCNN + GLCC
    _accuracy_evaluation(  conf_names[1] )
