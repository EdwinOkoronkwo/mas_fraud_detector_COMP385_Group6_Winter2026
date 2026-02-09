from pydantic import BaseModel
from typing import Dict, Optional

class IngestionResult(BaseModel):
    status: str # e.g., "SUCCESS" or "FAILED"
    row_counts: Dict[str, int]
    db_path: str
    verification_msg: str