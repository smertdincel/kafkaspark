import os

from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql.functions import split, col, regexp_replace



# SparkSession başlat
spark = SparkSession.builder \
    .appName("KafkaConsumerSpark") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0") \
    .getOrCreate()


# Kafka'dan veri okuyacak yapı
kafka_stream = (spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "test_topic") \
    .option("startingOffsets", "earliest")
                .load())

# Kafka mesajlarını işleme (CSV formatına uygun parçalama)
schema = StructType([
    StructField("total_bill", StringType(), True),
    StructField("tip", StringType(), True),
    StructField("sex", StringType(), True),
    StructField("day", StringType(), True),
    StructField("size", StringType(), True),
])

# Mesajları DataFrame'e dönüştür
stream_df = kafka_stream.selectExpr("CAST(value AS STRING)") \
    .select(split(col("value"), ",").alias("data")) \
    .select(
        col("data")[0].alias("total_bill"),
        col("data")[1].alias("tip"),
        col("data")[2].alias("sex"),
        col("data")[3].alias("day"),
        col("data")[4].alias("size")
    )

# # 'tip' sütunundaki virgülleri kaldırma ve sayıya dönüştürme
df = stream_df.withColumn("tip", regexp_replace("tip", ",", "").cast("int"))


# # 'total_bill' sütununu tam sayıya dönüştürme
df = df.withColumn("total_bill", col("total_bill").cast("int"))

model_path = os.path.join(os.getcwd(), "linear_regression_model")
model = PipelineModel.load(model_path)
predictions = model.transform(df)



# İşlenen veriyi console'a yazdır (test amaçlı)
query = stream_df.writeStream \
    .outputMode("append") \
    .format("console") \
    .start()

query.awaitTermination()
