from sqlalchemy import Column, Integer, Float, String, DateTime
from datetime import datetime
from data.database import Base


class InferenceResult(Base):
    __tablename__ = "inference_results"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    cc_num = Column(String)
    merchant = Column(String, nullable=True)
    city = Column(String, nullable=True)
    home_location = Column(String, nullable=True)  # This is the missing one!
    category = Column(String, nullable=True)

    # Raw & Normalized Agent Scores
    gold_score = Column(Float)
    neuro_raw = Column(Float)
    neuro_norm = Column(Float)
    cluster_raw = Column(Float)
    cluster_norm = Column(Float)

    # Final System Output
    final_mas_score = Column(Float)
    prediction_mode = Column(String)  # e.g., "GOLD_VETO", "CONSENSUS_BOOST"
    explanation = Column(String)
    actual_label = Column(Integer, nullable=True)  # For performance tracking