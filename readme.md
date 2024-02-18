# FastAPI Image Prediction

This is a FastAPI application for predicting Glaucoma in images using a TensorFlow Lite model.

## Setup

1. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

   Make sure to replace `requirements.txt` with the actual filename if you choose to create one.

2. Run the FastAPI application:

    ```bash
    uvicorn main:app --reload
    ```

   The application will be accessible at `http://127.0.0.1:8000`.

## Usage

1. Upload an image file for prediction:

    ```bash
    http POST http://127.0.0.1:8000/predict file@path/to/your/image.jpg
    ```

   Replace `path/to/your/image.jpg` with the actual path to the image file you want to test.

2. View the prediction result in the response.
3. You can test APIs at ```http://127.0.0.1:8000/docs```

## Additional Notes

- The application saves the uploaded image locally (optional). You may modify the code to customize the behavior.

- Ensure that the TensorFlow Lite model file (`GL_model-8808088694067036160_tflite_2024-01-21T15_44_28.861186Z_model.tflite`) is in the same directory as your FastAPI application.

- Customize and extend the application as needed for your project requirements.

