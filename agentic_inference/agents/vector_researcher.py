from autogen_agentchat.agents import AssistantAgent


class VectorResearcher:
    def __init__(self, model_client, vector_service):
        self.vector_service = vector_service

        # 1. Define the function
        # Add **kwargs here as a safety net for LangChain's 'config' injection
        async def query_policy_guidelines(query: str) -> str:
            """Searches bank policy. Recovers section headers if a specific match is missing."""
            retriever = vector_service.get_retriever(search_kwargs={"k": 2})
            docs = await retriever.ainvoke(query)
            if not docs:
                return "NO POLICY DATA FOUND."

            header = "BANKING OPERATIONAL POLICY (v2026.1)\n"
            return f"{header}\n" + "\n\n".join([d.page_content for d in docs])

        self.agent = AssistantAgent(
            name="Vector_Researcher",
            model_client=model_client,
            tools=[query_policy_guidelines],
            system_message="""[ROLE: COMPLIANCE EXPLAINER]
            You receive a Math Risk Score and transaction details. 
            Your job is to provide a human-readable explanation using corporate policy context.

            INSTRUCTIONS:
            1. If Neuro MSE is LOW (< 0.10): Explain that the behavior is mathematically normal and consistent with historical patterns.
            2. If Neuro MSE is HIGH (> 0.30): Explain that the transaction shows high-entropy anomalies matching fraud signatures.
            3. Use the POLICY_CONTEXT to mention specific rules (e.g., Merchant Category or Geographic limits).
            4. If SQL data is missing, rely on the MATH_CONTEXT provided in the prompt to explain the result.

            DO NOT return SQL error messages. Return a 1-2 sentence professional audit summary.
            """
        )
