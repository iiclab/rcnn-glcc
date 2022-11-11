from pathlib import Path

from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer

from saba_dataset import get_saba_dicts
import rcnn_glcc


def _training( conf_name:str='./conf/rcnn.yaml', glcc_on:bool=False ):
    # Check-up
    if 'rcnn' in conf_name:
        assert glcc_on == False
    if 'glcc' in conf_name:
        assert glcc_on == True

    # Configurations
    cfg = get_cfg()
    cfg.merge_from_file( conf_name )
    cfg.MODEL.GLCC_ON = glcc_on
    cfg.MODEL.GLCC_OUTPUT = False  
    Path( cfg.OUTPUT_DIR ).mkdir(parents=True,exist_ok=True)

    # Train engine
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

    # Auto-grad on at only GLCC module
    #for param in trainer.model.named_parameters():
    #    if 'glcc' in param[0]:
    #        param[1].requires_grad = True
    #    else:
    #        param[1].requires_grad = False
                
    # Training
    trainer.train()
    

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

    # Training
    #_training( conf_names[0], glcc_on=False )  # RCNN
    _training( conf_names[1], glcc_on=True )   # RCNN + GLCC
