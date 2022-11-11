from pathlib import Path
import cv2, random, json
import numpy as np
from tqdm import tqdm

from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer


def get_saba_dicts( data_path:str='./data/train' ) -> list:
    # Check-up
    assert Path(data_path).exists(), f'{data_path} cannot be found'
    
    # Parameters
    data_type = data_path.split('/')[-1]
    pimg = Path(data_path) / 'Images'
    pann = Path(data_path) / 'Annotations'    
    dataset_dicts = []  # output
    fn_json = Path(data_path) / (data_type + '.json')  # output
    is_saba_2022 = True if "2022" in data_path else False
    assert data_type=='train' or data_type=='test'
    assert pimg.exists() and pann.exists()

    # Load
    if fn_json.exists():
        with fn_json.open('r') as f:
            dataset_dicts = json.load( f )
        return dataset_dicts
    
    # Get image stems
    stems =  [ p.stem for p in  pimg.glob('*.jpg') ]

    # Indices of fish categories
    fish_classes = []
    for sname in stems:
        if 'masaba' in sname:
            fish_classes.append( 0 )
        elif 'goma' in sname:
            fish_classes.append( 1 )
        elif 'hybrid' in sname:
            fish_classes.append( 2 )
        else:
            raise ValueError()
    
    # Make COCO format: https://cocodataset.org/#format-data
    for idx, (sname, fcls) in enumerate( zip( tqdm(stems), fish_classes) ):
        # Get image information
        fn_img = str( pimg / (sname + '.jpg') )
        assert Path(fn_img).exists(), 'Error: {} cannot be found'.format(fn_img)
        h, w = cv2.imread(fn_img).shape[:2]

        # Get annotation
        fn_ann = pann / (sname + '.txt')
        with fn_ann.open() as f:
            lines = f.readlines()

        # Get annotations
        # saba_2017 and 2018, 5->Red
        # saba_2022, 1->goma, 3->red (goma), 2->masaba, 4->red (masaba)

        objs = []
        for l in lines:
            
            # category id and bounding box
            cid, x1, y1, x2, y2 = map( int, l.strip().split(',') )
            if is_saba_2022:
                assert cid in [1,2,3,4], 'Error: cid={}'.format(cid)  # 2022
            else:
                assert cid in [1,2,3,5], 'Error: cid={}'.format(cid)  # 2017, 2018
            
            # Categopry IDs are two classes, Red or Fish.
            if is_saba_2022:
                if cid == 3 or cid==4:
                    cid = 0  # Red line
                else:
                    cid = 1  # Fish

            else:  # 2017 and 2018 dataset
                if cid == 5:
                    cid = 0  # Red line
                else:
                    cid = 1  # Fish
                    
            # annotation object
            o = {
                "bbox": [x1, y1, x2, y2],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": cid
            }
            objs.append( o )
            
        # COCO format
        #record = dict(file_name=fn_img, image_id=idx, height=h, width=w )
        record = dict(file_name=fn_img, image_id=idx, height=h, width=w, fish_class=fcls )  # Add fish category
        record['annotations'] = objs
        dataset_dicts.append( record )

    # Save
    with fn_json.open('w') as f:
        json.dump( dataset_dicts, f )
        
    return dataset_dicts
    

        
if __name__ == '__main__':
    # Make json file for annotations
    if False:
        dataset_dicts = get_saba_dicts( './data/saba_20220930/train' )
        dataset_dicts = get_saba_dicts( './data/saba_20220930/test' )
        dataset_dicts = get_saba_dicts( './data/train' )
        dataset_dicts = get_saba_dicts( './data/test' )

    # Get averages of BGR
    # 2017 Mean: 41.36933469 31.64481072 29.04405737
    # 2017 STD:  6.31631883 4.92987802 4.89906234
    # 2022 Mean BGR =  [121.87789485 126.51371779 106.53628093]
    # 2022 STD BGR  =  [3.57814549 3.40182277 2.82945951]
    if True:
        db_dicts = get_saba_dicts('./data/saba_20220930/train')  # Get annotations
        vals = np.zeros( (len(db_dicts), 3) )  # output array
        for i, d in enumerate( tqdm(db_dicts) ):
            im = cv2.imread( d['file_name'] )  # image
            vals[i] = im.mean(axis=(0,1))  # Get averages
        print('Mean BGR = ', vals.mean( axis=0 ) )
        print('STD BGR = ', vals.std( axis=0 ) )

    # Test
    if False:
        # Register the balloon dataset to detectron2
        for d in ["train", "test"]:
            DatasetCatalog.register("saba_" + d, lambda d=d: get_saba_dicts(d) )
            MetadataCatalog.get("saba_" + d).set(thing_classes=['red','fish'])
        saba_metadata = MetadataCatalog.get("saba_train")

        # To verify the dataset is in correct format, let's visualize the annotations of randomly selected samples in the training set:
        dataset_dicts = get_saba_dicts("test")
        for d in random.sample(dataset_dicts, 3):
            img = cv2.imread(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata=saba_metadata, scale=0.5)
            out = visualizer.draw_dataset_dict(d)
            cv2.imshow('image',out.get_image()[:, :, ::-1])
            cv2.waitKey(0)

        
