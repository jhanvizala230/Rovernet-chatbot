import cv2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode


# Your original metadata
metadata_json = [
    {"id": 1, "name": "sky", "supercategory": "object"},
    {"id": 2, "name": "ridge", "supercategory": "object"},
    {"id": 3, "name": "soil", "supercategory": "object"},
    {"id": 4, "name": "sand", "supercategory": "object"},
    {"id": 5, "name": "bedrock", "supercategory": "object"},
    {"id": 6, "name": "rock", "supercategory": "object"},
    {"id": 7, "name": "rover", "supercategory": "object"},
    {"id": 8, "name": "trace", "supercategory": "object"},
    {"id": 9, "name": "hole", "supercategory": "object"}
]

thing_classes = [item["name"] for item in sorted(metadata_json, key=lambda x: x["id"])]
custom_metadata = {
    "thing_classes": thing_classes,
    "thing_colors": [
        (214, 39, 40),    # Red for sky
        (148, 103, 189),  # Purple for ridge
        (140, 86, 75),    # Brown for soil
        (227, 119, 194),  # Pink for sand
        (127, 127, 127),  # Gray for bedrock
        (188, 189, 34),   # Olive for rock
        (23, 190, 207),   # Cyan for rover
        (31, 119, 180),   # Blue for trace
        (255, 127, 14)    # Orange for hole
    ]
}

MetadataCatalog.get("terrain_test").set(**custom_metadata)

def load_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = "./model/model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 9
    cfg.INPUT.MIN_SIZE_TEST = 200
    cfg.INPUT.MAX_SIZE_TEST = 1500
    predictor = DefaultPredictor(cfg)
    return predictor

predictor = load_model()
metadata = MetadataCatalog.get("terrain_test")

def predict_image(image,original_filename):
    
    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")
    predictions = []
    for i in range(len(instances)):
        pred = {
            "class": instances.pred_classes[i].item(),
            "score": instances.scores[i].item(),
            "bbox": instances.pred_boxes[i].tensor.tolist()[0],
        }
        predictions.append(pred)

    v = Visualizer(image[:, :, ::-1],
                   metadata=metadata, 
                   scale=1.0, 
                   instance_mode=ColorMode.IMAGE_BW )
    
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    predicted_filename = f"./results/predicted/{original_filename}_predicted.jpg"
    cv2.imwrite(predicted_filename,out.get_image()[:, :, ::-1])
    return outputs, predictions, predicted_filename, metadata





