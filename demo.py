import os, cv2, random

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer

# GLCC
from saba_dataset import get_saba_dicts
import rcnn_glcc
from test import DefaultPredictor_GLCC

    

def _main_demo( conf_name:str='./conf/glcc_2022.yaml' ):
    assert 'glcc' in conf_name
    fish_names = ['masaba','goma']  # HARD CODED
    
    cfg = get_cfg()
    cfg.merge_from_file( conf_name )
    cfg.MODEL.GLCC_ON = True
    cfg.MODEL.GLCC_OUTPUT = True  # Get GLCC output
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    predictor = DefaultPredictor_GLCC(cfg)
    box_metadata = predictor.metadata
        
    # Show some results
    if '2017' in cfg.DATASETS.TEST[0]:
        data_path = './data/test'
    elif '2022' in cfg.DATASETS.TEST[0]:
        data_path = './data/saba_20220930/test'

    dataset_dicts = get_saba_dicts( data_path )
    for d in random.sample(dataset_dicts, 3):
        _fn = d["file_name"]
        im = cv2.imread( _fn )
        outputs, pred_fish = predictor(im)

        # Fish class
        if pred_fish is not None:
            pred_fish_name = fish_names[ pred_fish ]
        else:
            pred_fish_name = 'None'
        
        # Draw
        v = Visualizer( im[:, :, ::-1], metadata=box_metadata, scale=0.5 )
        out = v.draw_instance_predictions(outputs[0]["instances"].to("cpu"))
        title_str = 'GT:{}, Pred:{}'.format( fish_names[d['fish_class']], pred_fish_name )
        cv2.imshow( title_str, out.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyWindow('result')
        
    
    
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

    _main_demo( conf_names[1] ) 
