# from fastapi import FastAPI, UploadFile, File,Query
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
from backend.inference import predict_image
import cv2
from backend.information_extraction import save_rich_predictions, merge_images,encode_image_to_base64
from backend.prompt import generate_analytical_prompt
from  backend.ollama import ask_ollama, ask_ollama_streaming

# # Configure basic logging
# logger.basicConfig(
#     level=logger.INFO,  # DEBUG, INFO, WARNING, ERROR, CRITICAL
#     format='%(asctime)s - %(levelname)s - %(message)s',
# )

def predict(image, filename, question:str,stream : bool = False):
 
    # image = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    print(filename)
    original_path = f"./results/original_images/{filename}"
    cv2.imwrite(original_path, image) 

    outputs, predictions, predicted_filename, metadata = predict_image(image,filename)
    preds,output_json_file = save_rich_predictions(outputs,filename,metadata)
    output_path = merge_images(original_path,predicted_filename,filename)
    image_base_64 = encode_image_to_base64(output_path)
    prompt = generate_analytical_prompt(preds,question)

    if stream:
    # Return generator for streaming
        return ask_ollama_streaming(prompt, image_base_64), output_path
    else:
        # Return single response (backward compatible)
        response = ask_ollama(prompt, image_base_64)
        return response, output_path


    # response = ask_ollama(prompt,image_base_64,model="llava:7b")
    # print(f"Response : {response}")
    # return response,output_path
    # # return {"predictions": predictions}

if __name__ == "__main__":
    file_path = "D:\\RoverNet-main\\RoverNet-main\\test\images\\0003ML0000000110100031E01_DXXX.jpg"
    image = cv2.imread(file_path)
    filename= file_path.split("\\")[-1]
    question = "what are the classes that are detected. What are there confidence score and which class occupies maximum area"
    print("Starting streaming analysis...")
    response_stream, output_path = predict(image, filename, question, stream=True)
    
    for chunk in response_stream:
        print(chunk, end="", flush=True)
    
    print(f"\n\nAnalysis saved to: {output_path}")