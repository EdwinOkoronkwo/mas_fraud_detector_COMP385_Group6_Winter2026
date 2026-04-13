from autogen_agentchat.agents import AssistantAgent


class VectorResearcher:
    def __init__(self, model_client, vector_service):
        self.vector_service = vector_service

        # Define the tool with built-in redundancy
        async def query_policy_guidelines(query: str) -> str:
            """
            Searches bank policy v2026.1.
            If specific query fails, it pulls general fraud threshold guidelines.
            """
            try:
                retriever = self.vector_service.get_retriever(search_kwargs={"k": 2})
                docs = await retriever.ainvoke(query)

                # Fallback Logic: If specific search (e.g., "Neuro MSE 0.05") fails
                if not docs or len(docs) == 0:
                    fallback = "Standard Anomaly Thresholds and Category Risk Rules"
                    docs = await retriever.ainvoke(fallback)

                if not docs:
                    return "ERROR: Policy database unreachable or empty."

                header = "--- BANKING OPERATIONAL POLICY (v2026.1) ---\n"
                return header + "\n\n".join([d.page_content for d in docs])

            except Exception as e:
                return f"TECHNICAL ERROR DURING POLICY RETRIEVAL: {str(e)}"

        # Now attach the tool to the agent
        self.agent = AssistantAgent(
            name="Vector_Researcher",
            model_client=model_client,
            tools=[query_policy_guidelines],
            system_message="""[ROLE: COMPLIANCE ANALYST]
            1. Search the policy for rules matching the MSE score or Merchant Category.
            2. Extract the specific Rule ID and required compliance action.
            3. If the tool returns general guidelines, apply the 'Standard Threshold' rule."""
        )