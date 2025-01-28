import numpy as np
import onnxruntime as rt
from pyspark.ml.feature import StandardScalerModel
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("FraudDetection").getOrCreate()

# Load the test data and scaler
scaler_model = StandardScalerModel.load("artifact/scaler_model")
test_df = spark.read.parquet("artifact/test_data.parquet")

# Split the features and labels
X_test = np.array([row.features.toArray() for row in test_df.collect()])
y_test = np.array([row.label for row in test_df.collect()])

# Use the scaler model to transform the test data
scaled_test_df = scaler_model.transform(test_df)

# Extract scaled features and convert them to a NumPy array
X_test_scaled = np.array([row.scaled_features.toArray() for row in scaled_test_df.collect()])

# Load the ONNX model for inference
sess = rt.InferenceSession("models/fraud/1/model.onnx", providers=rt.get_available_providers())
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

# Make predictions
y_pred_temp = sess.run([output_name], {input_name: X_test_scaled.astype(np.float32)})
y_pred_temp = np.asarray(np.squeeze(y_pred_temp[0]))

# Apply threshold to predictions
threshold = 0.95
y_pred = np.where(y_pred_temp > threshold, 1, 0)

# Show the results:
from sklearn.metrics import precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

y_test_arr = y_test.squeeze()
correct = np.equal(y_pred, y_test_arr).sum().item()
acc = (correct / len(y_pred)) * 100
precision = precision_score(y_test_arr, np.round(y_pred))
recall = recall_score(y_test_arr, np.round(y_pred))

print(f"Eval Metrics: \n Accuracy: {acc:>0.1f}%, "
      f"Precision: {precision:.4f}, Recall: {recall:.4f} \n")

c_matrix = confusion_matrix(y_test_arr, y_pred)
ConfusionMatrixDisplay(c_matrix).plot()

# Example: Is Sally's transaction likely to be fraudulent?
#
# Here is the order of the fields from Sally's transaction details:
#
# distance_from_last_transaction
# ratio_to_median_price
# used_chip
# used_pin_number
# online_order

sally_transaction_details = [
    [0.3111400080477545,
     1.9459399775518593,
     1.0,
     0.0,
     0.0]
]

sally_df = spark.createDataFrame([(Vectors.dense(sally_transaction_details[0]),)], ["features"])

scaled_sally_df = scaler_model.transform(sally_df)

scaled_sally = np.array([row.scaled_features.toArray() for row in scaled_sally_df.collect()])

prediction = sess.run([output_name], {input_name: scaled_sally.astype(np.float32)})

print("Is Sally's transaction predicted to be fraudulent? (true = YES, false = NO) ")
print(np.squeeze(prediction) > threshold)

print("How likely was Sally's transaction to be fraudulent? ")
print("{:.5f}".format(100 * np.squeeze(prediction)) + "%")
