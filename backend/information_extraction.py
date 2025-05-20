import json
from datetime import datetime
from PIL import Image
# from logger import logger


def save_rich_predictions(outputs,image_name,metadata):
    # Extract all metadata
    preds = {
        "image": image_name.split(".")[0],
        "objects": [],
        "timestamp": datetime.now().isoformat()
    }
    # logger.info(preds["image"])
    
    for i, (class_id, score, box, mask) in enumerate(zip(
        outputs["instances"].pred_classes.cpu().numpy(),
        outputs["instances"].scores.cpu().numpy(),
        outputs["instances"].pred_boxes.tensor.cpu().numpy(),
        outputs["instances"].pred_masks.cpu().numpy()
    )):
        preds["objects"].append({
            "id": i,
            "class": metadata.thing_classes[class_id],
            "confidence": float(score),
            "bbox": box.tolist(),  # [x1, y1, x2, y2]
            "area": int(mask.sum()),
            "mask": mask.tolist()  # Optional: for detailed shape analysis
        })
    output_json_file = f"./results/predicted/{preds['image']}_pred.json"
    with open(output_json_file, "a") as f:
            json.dump(preds, f, indent=2)
    return preds,output_json_file
    

def merge_images(img1_path, img2_path,filename, output_path="merged.jpg"):
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    # Resize to same height
    if img1.size[1] != img2.size[1]:
        img2 = img2.resize((img2.size[0], img1.size[1]))
    combined = Image.new("RGB", (img1.width + img2.width, img1.height))
    combined.paste(img1, (0, 0))
    combined.paste(img2, (img1.width, 0))
    output_path = f"./results/predicted/filename_{output_path}"
    combined.save(output_path)
    return output_path

import base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
    


# get_prediction_desc(pred_data)

def generate_analytical_prompt(pred_data, question):
    """Generates a strict Llama 3.2 prompt for Mars terrain analysis"""
    # Calculate statistics
    class_details = get_prediction_desc(pred_data=pred_data)
    # Use triple quotes and proper line breaks
    prompt_template = """The description provides details about the segmented region obtained from
    the segmentation output using detectron2. alsong with this you also have original and segmented images available. 
    Left image is original and right is segmented. Based on description and images available answers user questions
    {object_details}

    ## Rules
    1. Answer ONLY with the detected classes below.
    2. Provide Your view based on the image you are observing.
    3. Optimize the answers and dont repeat infomation
    {question}
    
    Assistant: """

    return prompt_template.format(
        object_details="\n".join(class_details),
        question=question)