import numpy as np
import onnxruntime as rt
import uvicorn
from fastapi import FastAPI, Request
from pyspark.ml.feature import StandardScalerModel
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession

app = FastAPI()

# Initialize Spark session and load models on startup
spark = SparkSession.builder.appName("FraudDetection").getOrCreate()
scaler_model = StandardScalerModel.load("../train/artifact/scaler_model")
# Initialize the ONNX runtime session (using available providers)
sess = rt.InferenceSession("../train/models/fraud/1/model.onnx", providers=rt.get_available_providers())
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name


# Health endpoint required by KServe
@app.get("/healthz")
def healthz():
    return {"status": "ok"}


# Predict endpoint following KServe's custom predictor spec:
@app.post("/v1/models/fraud:predict")
async def predict(request: Request):
    """
    Predict endpoint.

    To test the custom predictor running on localhost, you can use the following curl command:

    curl -X POST "http://localhost:8080/v1/models/fraud:predict" \
      -H "Content-Type: application/json" \
      -d '{"instances": [[0.31114, 1.94594, 1.0, 0.0, 0.0]]}'

    This command sends a POST request to the '/v1/models/fraud:predict' endpoint with a JSON payload.
    The payload should include an "instances" key, where the value is a list of feature vectors.

    Each feature vector must contain the following values in order:
        1. distance_from_last_transaction: a float representing the distance from the last transaction.
        2. ratio_to_median_price: a float indicating the ratio of the transaction amount to the median price.
        3. used_chip: a binary indicator (1.0 for true, 0.0 for false) if a chip was used.
        4. used_pin_number: a binary indicator (1.0 for true, 0.0 for false) if a PIN was used.
        5. online_order: a binary indicator (1.0 for true, 0.0 for false) if the order was made online.

    The server responds with a JSON object that includes:
        - "raw": the raw probability output from the ONNX model.
        - "predictions": the binary prediction (0 or 1) after applying the defined threshold.
    """

    # Expecting a JSON payload with a key "instances"
    data = await request.json()
    instances = data.get("instances")
    if instances is None:
        return {"error": "No instances provided. Expecting a key 'instances' with list of feature vectors."}

    # Create a Spark DataFrame from the input instances. Each instance should be a list of floats.
    rows = [(Vectors.dense(features),) for features in instances]
    df = spark.createDataFrame(rows, ["features"])

    # Apply the Spark scaler transformation
    scaled_df = scaler_model.transform(df)
    # Collect the scaled features as a NumPy array
    x_scaled = np.array([row.scaled_features.toArray() for row in scaled_df.collect()])

    # Run the ONNX model for inference
    predictions = sess.run([output_name], {input_name: x_scaled.astype(np.float32)})
    predictions = np.squeeze(np.asarray(predictions[0]))

    # Apply the threshold to get binary predictions (adjust threshold as needed)
    threshold = 0.95
    predictions = (predictions > threshold).astype(int)

    # Return both the raw predictions and the thresholded binary predictions
    return {"predictions": predictions.tolist(), "raw": predictions.tolist()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
