from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
import base64 
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os
# from lanchain_community.tools.wolfram_alpha import WolframAlphaQueryRun

load_dotenv(r"C:\Projects\RAG_PoC\agents_course_hf\.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# search_tool = DuckDuckGoSearchRun()
search = DuckDuckGoSearchAPIWrapper()

def search_function(query: str) -> str:
    """Search the web for information."""
    try:
        results = search.run(query)
        if not results or results.strip() == "":
            return "No search results found. Please try a different query."
        return results
    except Exception as e:
        return f"Error during search: {str(e)}"


search_tool = Tool(
    name="duckduckgo_search_tool",
    description="Search the web for infromation about recent events, facts, or anything that requires up-to-date information.",
    func=search_function
)

vision_llm = ChatOpenAI(model="gpt-4o")

def image_describer(image_path: str) -> str:
    """Describes the content of an image."""

    description = ""

    try:
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        message = [
            HumanMessage(
                content=[
                    {
                    "type": "text",
                    "text": (
                        "Describe the type of image you see, if it is a photo, a drawing, a painting, etc. "
                        "Then describe the content of the image in the most detailled way possible. "
                        "You will start by describing the front of the image, then the back of the image if possible. "
                        "If the image contains text, you will extract it and describe it in the most detailler way possible. "
                        "If the image is a document, you will extract the text. Return only the text in this case, no explanations."
                        
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                        }
                    }
                ]
            )
        ]

        # call the vision model
        response = vision_llm(message)
        description += response.content + "\n\n"

        return description.strip()

    except Exception as e:
        print(f"Error reading image file: {e}")
        return "Error reading image file."




analyze_image_tool = Tool(
    name="analyze_image_tool",
    func=image_describer,
    description="Analyzes an image and returns a detailled description of it."
)

def analyze_text(text: str) -> str:
    """Analyzes a text and returns a detailled description of it."""
    return ""

# def math_tool():


#     return

