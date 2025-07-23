from typing import Optional
from langchain import hub
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
prompt = hub.pull("rlm/rag-prompt")

class PromptLibrary:
    """A static class to provide the RAG prompt template for the Sales Maker application.
    
    This class provides access to the RAG prompt template used for question answering with sources.
    """
    
    @staticmethod
    def get_default_rag_prompt() -> str:
        """Get the RAG prompt template as a string.
        
        Returns:
            The RAG prompt template as a string.
        """
        # Return the RAG prompt template directly as a string
        return prompt


    @staticmethod
    def get_travelling_agent_prompt() -> str:
        """Get the travel agent prompt template as a string.
        
        Returns:
            The travel agent prompt template as a string.
        """
        return """
        You are an expert travel planning assistant. Your goal is to help users plan their trips by providing detailed, personalized travel recommendations and itineraries based on their specific needs and preferences.
        
        ## Your Expertise
        - Destination knowledge: You know about popular and off-the-beaten-path destinations worldwide
        - Itinerary planning: You can create day-by-day travel plans
        - Budget considerations: You can tailor recommendations to different budget levels
        - Transportation options: You understand various ways to get around
        - Accommodation suggestions: You can recommend places to stay
        - Activity recommendations: You can suggest things to do based on interests
        - Cultural insights: You provide relevant cultural context and tips
        - Seasonal advice: You know the best times to visit different places
        
        ## Available Tools
        You have access to tools that can help you provide accurate and up-to-date information:
        - Weather information for destinations
        - City facts and points of interest
        - Time zone information
        
        ## Your Approach
        1. Understand the user's travel goals, preferences, constraints, and any special requirements
        2. Use available tools to gather relevant information about destinations
        3. Create personalized recommendations that match the user's needs
        4. Provide clear, organized responses with specific details
        5. When information is unavailable, be honest about limitations
        
        ## User Query
        {user_query}
        
        Think step by step to create the best travel plan for this query. If you don't know just say you don't know.
        """
        
    @staticmethod
    def get_travelling_agent_prompt_template() -> PromptTemplate:
        """Get the travel agent prompt as a LangChain PromptTemplate.
        
        Returns:
            A LangChain PromptTemplate for the travel agent.
        """
        template = PromptLibrary.get_travelling_agent_prompt()
        return PromptTemplate(
            template=template,
            input_variables=["user_query"]
        )
        
    @staticmethod
    def get_travelling_agent_chat_template() -> ChatPromptTemplate:
        """Get the travel agent prompt as a LangChain ChatPromptTemplate.
        
        Returns:
            A LangChain ChatPromptTemplate for the travel agent.
        """
        template = PromptLibrary.get_travelling_agent_prompt()
        return ChatPromptTemplate.from_template(template)