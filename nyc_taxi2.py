
# Importing required modules
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, hour
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Creating Spark session
spark = SparkSession.builder \
    .appName("NYC Taxi Trip Analysis") \
    .getOrCreate()

# Loading dataset
data = spark.read.csv("Big data/yellow_tripdata_2015-01.csv", header=True, inferSchema=True)

# Converting pickup and dropoff times to timestamp
data = data.withColumn("tpep_pickup_datetime", to_timestamp("tpep_pickup_datetime")) \
           .withColumn("tpep_dropoff_datetime", to_timestamp("tpep_dropoff_datetime"))

# Converting trip duration in minutes
data = data.withColumn("trip_duration",
                       (col("tpep_dropoff_datetime").cast("long") -
                        col("tpep_pickup_datetime").cast("long")) / 60)

# Extract hour of the day
data = data.withColumn("hour_of_day", hour("tpep_pickup_datetime"))

# Filtering invalid or extreme values
data = data.filter((col("trip_distance") > 0) &
                   (col("trip_distance") <= 100) &
                   (col("trip_duration") > 0))

# Select relevant numerical columns for correlation
numeric_cols = ["trip_distance", "trip_duration", "hour_of_day", "fare_amount"]

# Create an empty dictionary to store correlations
corr_dict = {}

# Compute correlations between each pair of columns
for col1 in numeric_cols:
    corr_dict[col1] = []
    for col2 in numeric_cols:
        corr_value = data.stat.corr(col1, col2)
        corr_dict[col1].append(corr_value)

# Convert the correlation dictionary to a Pandas DataFrame
corr_matrix = pd.DataFrame(corr_dict, index=numeric_cols, columns=numeric_cols)

# Plot the correlation matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation Matrix")
plt.show()

# Sorting relevant columns for modeling
data = data.select("trip_distance", "trip_duration", "hour_of_day", "fare_amount")

# Split the data into training and test sets
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Combining features into a single vector
assembler = VectorAssembler(inputCols=["trip_distance", "trip_duration", "hour_of_day"],
                            outputCol="features")

train_data = assembler.transform(train_data)
test_data = assembler.transform(test_data)

# Training Linear Regression model
lr = LinearRegression(featuresCol="features", labelCol="fare_amount")
model = lr.fit(train_data)

# Make predictions on the test set
predictions = model.transform(test_data)

# Evaluate the model using RMSE and MAE
evaluator_rmse = RegressionEvaluator(labelCol="fare_amount", predictionCol="prediction", metricName="rmse")
rmse = evaluator_rmse.evaluate(predictions)

evaluator_mae = RegressionEvaluator(labelCol="fare_amount", predictionCol="prediction", metricName="mae")
mae = evaluator_mae.evaluate(predictions)

print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# Convert predictions to Pandas DataFrame for visualization
predictions_pd = predictions.select("fare_amount", "prediction").toPandas()

# Plot: Actual vs Predicted Fare Amount
plt.figure(figsize=(8, 6))
plt.scatter(predictions_pd["fare_amount"], predictions_pd["prediction"],
            c=abs(predictions_pd["fare_amount"] - predictions_pd["prediction"]),
            cmap='coolwarm', alpha=0.6)
plt.colorbar(label='Prediction Error')
plt.plot([0, 100], [0, 100], color='black', lw=2, linestyle='--')  # 45-degree line

plt.title("Actual vs Predicted Fare Amount (Filtered)")
plt.xlabel("Actual Fare Amount ($)")
plt.ylabel("Predicted Fare Amount ($)")
plt.xlim(0, 100)
plt.ylim(0, 100)

# Save the plot
plt.savefig("actual_vs_predicted_fare.png")
plt.show()

# Stop the Spark session
spark.stop()
