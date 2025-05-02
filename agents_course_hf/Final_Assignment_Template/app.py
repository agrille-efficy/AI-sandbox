import os
import gradio as gr
import requests
import inspect
import pandas as pd
import re
import json
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from tools import search_tool, analyze_image_tool
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, Optional
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.prebuilt import ToolNode, tools_condition


# (Keep Constants as is)
# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

# --- Basic Agent Definition ---
class AgentState(TypedDict):
    input_file: Optional[str]
    messages: Annotated[list[AnyMessage], add_messages]

class BasicAgent:
    def __init__(self):
        system_prompt = """
        You are a general AI assistant. I will ask you a question.

        First, explore your reasoning process step by step. Consider all relevant facts and possibilities.

        Then, provide your answer using EXACTLY this format:

        FINAL ANSWER: [Your concise answer]

        Your FINAL ANSWER should be:
        - For numbers: Just the number without commas or units (unless specified)
        - For text: As few words as possible with no articles or abbreviations 
        - For lists: Comma-separated values following the above rules

        Important: The evaluation system will ONLY read what comes after "FINAL ANSWER:". Make sure your answer is correctly formatted.
        """

        self.chat = ChatOpenAI(temperature=0)
        self.tools = [search_tool, analyze_image_tool]
        self.chat_with_tools = self.chat.bind_tools(self.tools)
        self.system_message = SystemMessage(content=system_prompt)
        print("BasicAgent initialized.")

    def __call__(self, question: str, task_id) -> str:
        """Processes the input question using the chat_with_tools object and returns the anwer""" 
        try:
            image_urls = re.findall(r'https?://\S+?(?:png|jpg|jpeg|gif)', question)

            if image_urls:
                print(f"Detected image URLs: {image_urls}")
                image_descriptions = []

                for url in image_urls:
                    try: 
                        image_description = self.analyze_image_tool(url)
                        image_descriptions.append(f"Image analysis: {image_description}")
                    except Exception as e: 
                        image_descriptions.append(f"Couldn't analyze image {url}: {e}")

                enhanced_question = question + "\n\n" + "\n".join(image_descriptions)
                human_message = HumanMessage(content=enhanced_question)
            else:
                if any(keyword in question.lower() for keyword in ["how many", "who", "what", "when", "where", "which"]):
                    # For factual questions, add specific instructions to use tools
                    question = f"This is a factual question that requires using the search tool to find accurate information: {question}"
                    print("Added search instruction to question")

                human_message = HumanMessage(content=question)

            messages = [
                self.system_message,
                human_message
            ]

            response = self.chat_with_tools.invoke(messages)

            content = response.content or ""

            # Handle tool calls if present
            if hasattr(response, 'additional_kwargs') and 'tool_calls' in response.additional_kwargs:
                tool_calls = response.additional_kwargs['tool_calls']
                print(f"Tool call detected: {tool_calls}")
                
                # Process tool calls and add results to messages
                tool_results = []
                for tool_call in tool_calls:
                    tool_name = tool_call.get('function', {}).get('name')
                    if not tool_name:
                        continue
                        
                    # Extract the arguments
                    tool_args_str = tool_call.get('function', {}).get('arguments', '{}')
                    try:
                        tool_args = json.loads(tool_args_str)
                        
                        # Execute the appropriate tool
                        if tool_name == "duckduckgo_search" and 'query' in tool_args:
                            result = search_tool.func(tool_args['query'])
                            tool_results.append(f"Search result: {result}")
                        elif tool_name == "analyze_image" and 'image_url' in tool_args:
                            result = analyze_image_tool.func(tool_args['image_url'])
                            tool_results.append(f"Image analysis: {result}")
                            
                    except json.JSONDecodeError:
                        print(f"Could not parse tool arguments: {tool_args_str}")
                
                # If we have tool results, send a follow-up to process them
                if tool_results:
                    follow_up = f"Based on these tool results, please answer the original question:\n\n{question}\n\nTool Results:\n" + "\n".join(tool_results)
                    follow_up_message = HumanMessage(content=follow_up)
                    messages.append(follow_up_message)
                    response = self.chat_with_tools.invoke(messages)
                    content = response.content or ""

            # task_id_match = re.search(r"task_id:\s*(\w+)", question, re.IGNORECASE)
            # task_id = task_id_match.group(1) if task_id_match else "unknown_task_id"

            final_answer_match = re.search(r"FINAL ANSWER:\s*(.*?)(?:\n|$)", content, re.IGNORECASE | re.DOTALL)
            model_answer = final_answer_match.group(1).strip() if final_answer_match else "No answer found."

            reasoning = content.split("FINAL ANSWER:")[0].strip() if "FINAL ANSWER:" in content else content.strip()

            formatted_response = {
                "task_id": task_id,
                "model_answer": model_answer,
                "reasoning_trace": reasoning
            }

            return json.dumps(formatted_response)
        
        except Exception as e:
            print(f"Error in BasicAgent call: {e}")
            return json.dumps({
                "task_id": "error",
                "model_answer": "Error processing the question",
                "reasoning_trace": f"Error: {str(e)}"
            })

        except Exception as e:
            print(f"Error in BasicAgent call: {e}")
            return "Error processing the question."
    

    def assistant(self, state: AgentState):
        return {
            "messages": [self.chat_with_tools.invoke(state["messages"])]
        }
    
    def graph_builder(self):
        # Graph
        builder = StateGraph(AgentState)

        def enhanced_assistant(state):
            try:
                messages = [self.system_message] + state["messages"]
                response = self.chat_with_tools.invoke(messages)
                return {"messages": [response]}
            except Exception as e: 
                print(f"Error in assistant node: {e}")
                error_message = AIMessage(content=f"I'm sorry, I encountered an error: {e}")
                return {"messages": [error_message]}
            
        # Nodes
        builder.add_node("assistant", enhanced_assistant)
        builder.add_node("tools", ToolNode(self.tools))

        # Edges
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges(
            "assistant", 
            tools_condition, 
            {
                "tools": "tools", 
                "end": END
                }
            )
        
        builder.add_edge("tools", "assistant")
        agent = builder.compile()

        return agent
    

def run_and_submit_all( profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the BasicAgent on them, submits all answers,
    and displays the results.
    """
    # --- Determine HF Space Runtime URL and Repo URL ---
    space_id = os.getenv("SPACE_ID") # Get the SPACE_ID for sending link to the code

    if profile:
        username= f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate Agent ( modify this part to create your agent)
    try:
        agent = BasicAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None
    # In the case of an app running as a hugging Face space, this link points toward your codebase ( usefull for others so please keep it public)
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
             print("Fetched questions list is empty.")
             return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
         print(f"Error decoding JSON response from questions endpoint: {e}")
         print(f"Response text: {response.text[:500]}")
         return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # 3. Run your Agent
    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        try:
            # submitted_answer = agent(question_text, task_id)
            # answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            # results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})

            # Parse the JSON returned by your agent
            submitted_answer_json = agent(question_text, task_id)
            # answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer_json})
            # results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer_json})

            try:
                # Parse the JSON string to a dictionary
                answer_dict = json.loads(submitted_answer_json)
                answers_payload.append(answer_dict)  # Add the dictionary directly
                results_log.append({
                    "Task ID": task_id, 
                    "Question": question_text, 
                    "Submitted Answer": answer_dict.get("model_answer", "No answer found.")
                })
            except json.JSONDecodeError:
                print(f"Error parsing JSON for task {task_id}: {submitted_answer_json}")
                results_log.append({
                    "Task ID": task_id, 
                    "Question": question_text, 
                    "Submitted Answer": "ERROR: Invalid JSON response"
                })
        except Exception as e:
             print(f"Error running agent on task {task_id}: {e}")
             results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Prepare Submission 
    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}    
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    # 5. Submit
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df


    
# --- Build Gradio Interface using Blocks ---
with gr.Blocks() as demo:
    gr.Markdown("# Basic Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**

        1.  Please clone this space, then modify the code to define your agent's logic, the tools, the necessary packages, etc ...
        2.  Log in to your Hugging Face account using the button below. This uses your HF username for submission.
        3.  Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.

        ---
        **Disclaimers:**
        Once clicking on the "submit button, it can take quite some time ( this is the time for the agent to go through all the questions).
        This space provides a basic setup and is intentionally sub-optimal to encourage you to develop your own, more robust solution. For instance for the delay process of the submit button, a solution could be to cache the answers and submit in a seperate action or even to answer the questions in async.
        """
    )

    gr.LoginButton()

    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    # Removed max_rows=10 from DataFrame constructor
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    # Check for SPACE_HOST and SPACE_ID at startup for information
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID") # Get SPACE_ID at startup

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup: # Print repo URLs if SPACE_ID is found
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-"*(60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for Basic Agent Evaluation...")
    demo.launch(debug=True, share=False)