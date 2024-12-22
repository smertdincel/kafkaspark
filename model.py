from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.sql.functions import regexp_replace, col
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt
import os

def train():
    # SparkSession başlatma
    spark = SparkSession.builder.appName("SparkMLlibProject").getOrCreate()
    print("Spark session started successfully.")

    # CSV dosyasını yükleme
    df = spark.read.csv("tips.csv", header=True, inferSchema=True)

    # 'tip' sütunundaki virgülleri kaldırma ve sayıya dönüştürme
    df = df.withColumn("tip", regexp_replace("tip", ",", "").cast("int"))

    # 'total_bill' sütununu tam sayıya dönüştürme
    df = df.withColumn("total_bill", col("total_bill").cast("int"))

    # String sütunları indeksleme
    indexer = StringIndexer(inputCols=["sex", "day", "size"],
                            outputCols=["sex_index", "day_index", "size_index"],
                            handleInvalid="keep")

    # Özellikleri vektörleştirme
    assembler = VectorAssembler(
        inputCols=["sex_index", "day_index", "size_index", "tip"],
        outputCol="features",
        handleInvalid="skip"
    )

    # Model Tanımlama (Linear Regression)
    lr = LinearRegression(featuresCol="features", labelCol="total_bill", maxIter=10)

    # Pipeline oluşturma
    pipeline = Pipeline(stages=[indexer, assembler, lr])

    # Veriyi eğitim ve test olarak ayırma
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

    # Modeli eğitme
    model = pipeline.fit(train_data)

    # Modeli belirtilen dizine kaydet
    model_path = os.path.join(os.getcwd(), "linear_regression_model")
    model.write().overwrite().save(model_path)
    print(f"Model saved at: {model_path}")

    # Test verisini kullanarak tahmin yapma
    predictions = model.transform(test_data)

    # Tahminleri gösterme
    predictions.select("features", "total_bill", "prediction").show(5)

    # Tahminleri değerlendirme
    evaluator = RegressionEvaluator(labelCol="total_bill", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print(f"Root Mean Square Error (RMSE): {rmse}")

    # Görselleştirme (Gerçek değerler ve tahminler)
    pdf = predictions.select("total_bill", "prediction").toPandas()

    plt.figure(figsize=(10, 6))
    plt.scatter(pdf["total_bill"], pdf["prediction"], alpha=0.6, color="b", label="Predictions")
    plt.plot([pdf["total_bill"].min(), pdf["total_bill"].max()],
             [pdf["total_bill"].min(), pdf["total_bill"].max()], 'r--', label="Ideal Fit")
    plt.title("Linear Regression Predictions vs Actual Values")
    plt.xlabel("Actual Total Bill")
    plt.ylabel("Predicted Total Bill")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train()
