import re



class FraudCritic:
    def __init__(self, model_client):
        self.client = model_client
        # Define a high-fidelity mapping based on your Handbook's severity
        self.severity_weights = {
            "CRITICAL": 0.95,  # Immediate match for Policy GEO-001 High-Risk or NET-004
            "HIGH_RISK": 0.80,  # VEL-002 Rapid Fire or failed CVVs
            "MEDIUM_RISK": 0.50,  # Standard deviation outliers (AMT-003)
            "LOW_RISK": 0.20,  # Minor policy deviations
            "SAFE": 0.0  # No policy violations found
        }

    async def evaluate_rag_evidence(self, rag_summary: str) -> dict:
        prompt = f"""
        EXAMINATION TASK:
        You are the Lead Fraud Critic. Analyze the RAG summary below and assign a 
        severity tag from: {list(self.severity_weights.keys())}.

        RAG SUMMARY: 
        {rag_summary}

        INSTRUCTION: 
        If the summary mentions 'Mandatory MFA' or 'Nigeria/North Korea', use CRITICAL.
        If it mentions 'failed CVVs' or 'Rapid Fire', use HIGH_RISK.
        Otherwise, use SAFE unless a specific outlier is mentioned.

        RESPONSE FORMAT:
        Category: [TAG]
        Reasoning: [1 SENTENCE]
        """

        response = await self.client.create_completion(prompt)
        final_score = self._map_to_numeric(response.content)

        return {
            "policy_score": final_score,
            "critic_reasoning": response.content
        }

    def _map_to_numeric(self, llm_response: str) -> float:
        """
        Extracts the highest risk category found in the LLM text
        and maps it to a float.
        """
        normalized_response = llm_response.upper()

        # We look for the most severe tag present in the reasoning
        detected_scores = [
            weight for tag, weight in self.severity_weights.items()
            if tag in normalized_response
        ]

        # Default to 0.0 if Mistral hallucinates a tag not in our rubric
        return max(detected_scores) if detected_scores else 0.0