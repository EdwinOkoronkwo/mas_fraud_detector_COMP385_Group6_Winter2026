from autogen_agentchat.agents import AssistantAgent
from autogen_ext.tools.langchain import LangChainToolAdapter # Import the adapter
from langchain.tools import tool


from autogen_agentchat.agents import AssistantAgent
from autogen_ext.tools.langchain import LangChainToolAdapter

from langchain_core.tools import tool as langchain_tool # Use LangChain's decorator
from autogen_ext.tools.langchain import LangChainToolAdapter

from langchain_core.tools import tool as langchain_tool
from autogen_ext.tools.langchain import LangChainToolAdapter
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
            system_message="""You are the Compliance Scout. 
                   STRICT LIMITATION: You have NO personal knowledge. Your 'brain' is the 'query_policy_guidelines' tool.
                   1. To answer ANY question, you MUST call 'query_policy_guidelines'. 
                   2. NEVER apologize or provide general advice. 
                   3. If the tool returns nothing, state: 'NO POLICY DATA FOUND.' 
                   4. Cite the POLICY_CODE (e.g., GEO-001) in your response.
                   """
        )

# class VectorResearcher:
#     def __init__(self, model_client, vector_service):
#         self.vector_service = vector_service
#
#         # 1. Define the function
#         async def query_policy_guidelines(query: str) -> str:
#             """Searches bank policy. Recovers section headers if a specific match is missing."""
#             try:
#                 retriever = self.vector_service.get_retriever(search_kwargs={"k": 2})
#                 docs = await retriever.ainvoke(query)
#
#                 if not docs:
#                     docs = await retriever.ainvoke("POLICY_CODE section titles and general guidelines")
#
#                 if not docs:
#                     return "NO POLICY DATA FOUND."
#
#                 header = "BANKING OPERATIONAL POLICY: FRAUD RED FLAGS & ESCALATION (v2026.1)\n"
#                 content = "\n\n".join([d.page_content for d in docs])
#                 return f"{header}\n{content}"
#             except Exception as e:
#                 return f"Error accessing Policy DB: {str(e)}"
#
#         # 2. CONVERT to a LangChain BaseTool first
#         # This solves the "Expected BaseTool, got Coroutine" error
#         lc_tool = langchain_tool(query_policy_guidelines)
#
#         # 3. ADAPT for AutoGen
#         vector_tool = LangChainToolAdapter(lc_tool)
#
#         self.agent = AssistantAgent(
#             name="Vector_Researcher",
#             model_client=model_client,
#             tools=[vector_tool],
#             system_message="""You are the Compliance Scout.
#                    Return ONLY the single most relevant policy rule.
#                    CRITICAL: You DO NOT have personal knowledge.
#                    To answer ANY question, you MUST use the 'query_policy_guidelines' tool.
#                    If the tool returns no results, say 'No policy found.' Do not give general advice.
#                    If no exact rule exists, state 'NO SPECIFIC POLICY' and stop.
#                    ou are the Compliance Scout.
#                    STRICT LIMITATION: You have NO personal knowledge. Your 'brain' is the 'query_policy_guidelines' tool.
#                    1. To answer ANY question, you MUST call 'query_policy_guidelines'.
#                    2. NEVER apologize. NEVER say 'I'm here to help but...'
#                    3. If the tool returns nothing, simply say: 'NO POLICY DATA FOUND.'
#                    4. Do NOT give general advice. Only return the literal policy text from your tool
#                                """
#         )
# class VectorResearcher:
#     def __init__(self, model_client, vector_service):
#         self.vector_service = vector_service
#
#         @tool
#         def query_policy_guidelines(query: str) -> str:
#             """Searches internal bank policy for a single, most relevant fraud rule."""
#             try:
#                 # 1. Get retriever with top-k limit (k=1 for highest precision)
#                 # This ensures we only get the 'Top Signal' chunk.
#                 retriever = self.vector_service.get_retriever(
#                     search_kwargs={"k": 1}
#                 )
#                 docs = retriever.invoke(query)
#
#                 if not docs:
#                     return f"Policy DB: No direct match for '{query}'. Escalating to lead."
#
#                 # 2. Extract content and truncate if necessary (though k=1 usually fits)
#                 content = docs[0].page_content
#                 if len(content) > 600:
#                     content = content[:600] + "... [truncated for brevity]"
#
#                 return f"RELEVANT POLICY FOUND:\n{content}"
#             except Exception as e:
#                 return f"Policy DB Error: {str(e)}"
#
#         # 3. Adapter for AutoGen 0.4
#         vector_tool = LangChainToolAdapter(query_policy_guidelines)
#
#         self.agent = AssistantAgent(
#             name="Vector_Researcher",
#             model_client=model_client,
#             tools=[vector_tool],
#             system_message="""You are the Compliance Scout.
#             Return ONLY the single most relevant policy rule.
#             CRITICAL: You DO NOT have personal knowledge.
#             To answer ANY question, you MUST use the 'query_policy_guidelines' tool.
#             If the tool returns no results, say 'No policy found.' Do not give general advice.
#             If no exact rule exists, state 'NO SPECIFIC POLICY' and stop.
#             ou are the Compliance Scout.
#             STRICT LIMITATION: You have NO personal knowledge. Your 'brain' is the 'query_policy_guidelines' tool.
#             1. To answer ANY question, you MUST call 'query_policy_guidelines'.
#             2. NEVER apologize. NEVER say 'I'm here to help but...'
#             3. If the tool returns nothing, simply say: 'NO POLICY DATA FOUND.'
#             4. Do NOT give general advice. Only return the literal policy text from your tool
#                         """
#         )