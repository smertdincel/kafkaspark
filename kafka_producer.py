import time

from kafka import KafkaProducer
import pandas as pd
import json

# Kafka Producer bağlantısını oluştur

producer = KafkaProducer(bootstrap_servers='localhost:9092')  # Burada kendi Kafka sunucu adresini kullan
time.sleep(20)
# CSV dosyasını oku
csv_file_path = 'tips.csv'  # CSV dosyanın burada
data = pd.read_csv(csv_file_path)

# Tüm satırları döngüyle Kafka'ya gönder
for index, row in data.iterrows():
    try:
        # Satırı JSON formatına dönüştür
        message = json.dumps(row.to_dict()).encode('utf-8')  # CSV satırını JSON'e çevir
        # Kafka'ya gönder
        producer.send('test_topic', message)  # 'your_topic' yerine kendi topic adını yaz
        print(f"Sent: {message}")
    except Exception as e:
        print(f"Failed to send message at index {index}: {e}")







