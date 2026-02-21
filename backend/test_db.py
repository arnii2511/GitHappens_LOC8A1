import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()

print("DATABASE_URL =", os.getenv("DATABASE_URL"))

conn = psycopg2.connect(os.getenv("DATABASE_URL"), sslmode="require")
print("CONNECTED OK")
conn.close()