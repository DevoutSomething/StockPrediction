from database import db_connection

engine = db_connection.engine
Base = db_connection.Base

print("Creating database tables...")
Base.metadata.create_all(bind=engine)
print("✅ Database setup complete!")
