import hashlib
from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Customer(Base):
    __tablename__ = "customers"

    id = Column(Integer, primary_key=True, index=True)
    cc_num = Column(String, unique=True, index=True)
    customer_name = Column(String)
    job = Column(String)
    home_lat = Column(Float)
    home_long = Column(Float)
    avg_txn_amt = Column(Float)
    risk_score = Column(Float, default=0.0)

    @property
    def photo_url(self):
        """Generates a fixed, unique avatar based on the customer name."""
        # Using a hash of the name as a seed for Pravatar
        seed = hashlib.md5(self.customer_name.encode()).hexdigest()
        return f"https://i.pravatar.cc/150?u={seed}"