from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import HUGGINGFACE_API_KEY, MODEL_NAME


# Define tools
@tool
def get_glucose_predictions() -> list[float]:
    """Get glucose predictions from the model"""
    return [120, 115, 90, 75, 80, 110, 95, 68, 110, 125]


@tool
def analyze_low_glucose_risk(predictions: list[float]) -> dict:
    """Analyze risk of low glucose by detecting outliers"""
    predictions_array = np.array(predictions)
    mean = np.mean(predictions_array)
    std = np.std(predictions_array)

    low_threshold = mean - 2 * std
    low_glucose_values = [v for v in predictions if v < low_threshold]

    return {
        "mean_glucose": float(mean),
        "std_dev": float(std),
        "low_threshold": float(low_threshold),
        "low_glucose_count": len(low_glucose_values),
        "low_glucose_values": low_glucose_values,
        "risk_level": "high" if len(low_glucose_values) > 2 else "normal"
    }


def test_model(query: str) -> str:
    """Test function to invoke the model directly"""
    llm = HuggingFaceEndpoint(
        repo_id=MODEL_NAME,
        huggingfacehub_api_token=HUGGINGFACE_API_KEY,
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
)

    chat_model = ChatHuggingFace(llm=llm)
    messages: list[BaseMessage] = [HumanMessage(content=query)]
    response = llm.invoke(messages)
    return str(response)



class DataAnalyst:
    """A glucose data analyst agent that uses tools to analyze glucose predictions"""

    def __init__(self):
        """Initialize the DataAnalyst with HuggingFace"""
        self.tools = [get_glucose_predictions, analyze_low_glucose_risk]
        self.tools_dict = {tool.name: tool for tool in self.tools}

        # Initialize HuggingFace model
        llm = HuggingFaceEndpoint(
            repo_id=MODEL_NAME,
            huggingfacehub_api_token=HUGGINGFACE_API_KEY,
            temperature=0.0
        )
        self.model = ChatHuggingFace(llm=llm)
        # Bind tools to the model
        self.model_with_tools = self.model.bind_tools(self.tools)

    def invoke(self, query: str) -> str:
        """
        Execute the agent with the given query.

        Args:
            query: String input for the agent to process

        Returns:
            The agent's output as a string
        """
        messages: list[BaseMessage] = [HumanMessage(content=query)]
        response = self.model_with_tools.invoke(messages)
        messages.append(response)

        max_iterations = 3
        iteration = 0

        while hasattr(response, "tool_calls") and response.tool_calls and iteration < max_iterations:
            iteration += 1
            tool_results = []
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_input = tool_call["args"]

                tool = self.tools_dict.get(tool_name)
                if tool:
                    result = tool.invoke(tool_input)
                else:
                    result = f"Unknown tool: {tool_name}"

                tool_results.append({
                    "type": "tool_result",
                    "tool_call_id": tool_call["id"],
                    "content": str(result),
                })

            for result in tool_results:
                messages.append(ToolMessage(
                    content=result["content"],
                    tool_call_id=result["tool_call_id"]
                ))

            response = self.model_with_tools.invoke(messages)
            messages.append(response)

        if hasattr(response, "content") and isinstance(response.content, str):
            return response.content
        return str(response)
