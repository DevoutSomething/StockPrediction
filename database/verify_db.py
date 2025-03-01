from database.db_connection import engine
from sqlalchemy import inspect

def check_tables():
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print("Existing tables in the database:", tables)

if __name__ == "__main__":
    check_tables()
