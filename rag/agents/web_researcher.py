from autogen_agentchat.agents import AssistantAgent
from autogen_ext.tools.langchain import LangChainToolAdapter
from langchain_community.tools.tavily_search import TavilySearchResults

from autogen_ext.tools.langchain import LangChainToolAdapter
from langchain_community.tools.tavily_search import TavilySearchResults


class WebResearcher:
    def __init__(self, model_client):
        # 1. Initialize the tool with strict limits (k=2 results)
        tavily_tool = TavilySearchResults(
            k=2,
            include_raw_content=False
        )

        # 2. Define a custom function to act as a "Gatekeeper"
        # In your Web_Researcher class
        async def search_web(query: str) -> str:
            # Setting k=1 for maximum precision and minimum bloat
            raw_data = await tavily_tool.ainvoke({"query": query})

            formatted_results = []
            # Using a list slice [0:1] ensures it doesn't crash if 0 results come back
            for item in raw_data[:1]:
                content = item.get("content", "")[:400]  # 400 chars is usually safe
                url = item.get("url", "No URL")
                formatted_results.append(f"TOP SIGNAL: {content}...\nSource: {url}")

            return "\n\n".join(formatted_results) if formatted_results else "No relevant info found."

        # 3. Use the agent with our custom 'gatekeeper' tool
        self.agent = AssistantAgent(
            name="Web_Researcher",
            model_client=model_client,
            tools=[search_web],  # Pass the function directly
            system_message="""You are the Intelligence Scout. 
            Provide ONLY the high-level facts about external threats. 
            Keep your responses under 100 words."""
        )