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


def get_prediction_desc(pred_data):
    class_stats = {}
    for obj in pred_data["objects"]:
        cls = obj["class"]
        if cls not in class_stats:
            class_stats[cls] = {
                "count": 0,
                "confidences": [],
                "areas": []
            }
        class_stats[cls]["count"] += 1
        class_stats[cls]["confidences"].append(obj["confidence"])
        class_stats[cls]["areas"].append(obj["area"])

    # Format class details
    class_details = []
    for cls, stats in class_stats.items():
        avg_conf = sum(stats["confidences"]) / stats["count"]
        total_area = sum(stats["areas"])
        class_details.append(
            f"- {cls}: {stats['count']} object(s), "
            f"avg confidence={avg_conf:.2f}, "
            f"total area={total_area} px²"
        )
    print(class_details)
    return class_details


# def generate_simple_prompt(question):
#     prompt_template = """The description provides details about the segmented region obtained from
#     the segmentation output using detectron2. alsong with this you also have original and segmented images available. 
#     Left image is original and right is segmented. Based on description and images available answers user questions
#     {object_details}

#     ## Rules
#     1. Answer ONLY with the detected classes below.
#     2. Provide Your view based on the image you are observing.
#     3. Optimize the answers and dont repeat infomation
#     {question}
    
#     Assistant: """

#     return prompt_template.format(
#             object_details="\n".join(),
#             question=question)
