from pydantic import BaseModel, Field
from typing import Literal

class AuditResult(BaseModel):
    audit_result: Literal["Compliant", "Non-Compliant"] = Field(description="The final status of the audit")
    primary_violation: str = Field(description="The rule code violated, or 'None'")
    risk_level: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"] = Field(description="The risk level determined by the audit")
    summary: str = Field(description="A brief summary for the analyst")