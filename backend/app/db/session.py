import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

# DATABASE_URL = os.getenv("DATABASE_URL")

# Manejo seguro para LAB: Fallback a SQLite si no hay configuración
# Esto permite levantar el entorno sin configurar Postgres
raw_db_url = os.getenv("DATABASE_URL")

if not raw_db_url:
    print("[WARN] DATABASE_URL no definida. Usando SQLite local (LAB mode).")
    DATABASE_URL = "sqlite:///./lab.db"
    connect_args = {"check_same_thread": False}  # Necesario para SQLite + FastAPI
else:
    DATABASE_URL = raw_db_url
    connect_args = {}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

# ✅ NUEVO: dependency-compatible helper
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
