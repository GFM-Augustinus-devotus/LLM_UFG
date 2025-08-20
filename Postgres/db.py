import os
from dotenv import load_dotenv, find_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker, declarative_base

load_dotenv(find_dotenv(filename=".env"), override=True)

PGHOST = os.getenv("PGHOST")
PGPORT_STR = os.getenv("PGPORT")

try:
    PGPORT = int(PGPORT_STR)
except(TypeError, ValueError):
    raise RuntimeError(f"PGPORT inválido: {PGPORT_STR!r}")

PGDATABASE = os.getenv("PGDATABASE")
PGUSER = os.getenv("PGUSER")
PGPASSWORD = os.getenv("PGPASSWORD")
ECHO = os.getenv("SQLALCHEMY_ECHO", "false").lower() == "true"

if not PGPASSWORD:
    raise RuntimeError("PGPASSWORD não definido no .env")

DATABASE_URL = URL.create(
    "postgresql+psycopg",
    username=PGUSER,
    password=PGPASSWORD,
    host=PGHOST,
    port=PGPORT,
    database=PGDATABASE
)

engine = create_engine(DATABASE_URL, echo=ECHO, future=True, pool_pre_ping=True)
Sessionlocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()