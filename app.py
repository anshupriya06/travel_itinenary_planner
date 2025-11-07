import os
import gradio as gr
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatMessagePromptTemplate, ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.graph import MermaidDrawMethod
from IPython.display import Image, display
from langchain_groq import ChatGroq

#Define the state structure
class PlannerState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], "Messages in the conversation"]
    city: str
    interests: List[str] # Changed to List[str]
    itinerary: str


# define the llm
llm = ChatGroq(
    temperature=0,
    groq_api_key = os.getenv("GROQ_API_KEY"),
    model_name = "llama-3.3-70b-versatile"
)

#define the itinerary prompt
itinerary_prompt = ChatPromptTemplate.from_messages([
    ("system","You are a helpful travel assistant. create a day trip itinerary for {city} based on the user's interests: {interests}.Provide a brief, bulleted itinery. "),
    ("human", "Create an itinerary for my day trip."),
])

# Define the workflow functions
def input_city(state: PlannerState) -> PlannerState:
    print("City received:", state["city"])
    return state


def input_interest(state: PlannerState) -> PlannerState:
    print("User interests:", state["interests"])
    return state

#

def create_itinerary(state: PlannerState) -> PlannerState:
  print(f"Creating an itinerary for {state['city']} based on interests: {', '.join(state['interests'])}")
  response = llm.invoke(
      itinerary_prompt.format(
          city=state["city"],
          interests=", ".join(state["interests"])
          )
      )

  itinerary_content = response.content
  print("\nGenerated Itinerary:\n", itinerary_content)
  
  return {
      **state,
      "messages": state["messages"] + [AIMessage(content=itinerary_content)],
      "itinerary": itinerary_content,
  }

# Build the LangGraph workflow
workflow = StateGraph(PlannerState)
workflow.add_node("input_city", input_city)
workflow.add_node("input_interest", input_interest)
workflow.add_node("create_itinerary", create_itinerary)

workflow.set_entry_point("input_city")
workflow.add_edge("input_city", "input_interest")
workflow.add_edge("input_interest", "create_itinerary")
workflow.add_edge("create_itinerary", END)

app = workflow.compile()

# Define the gradio appplication
def travel_planner(city: str, interests: str):
    print(f"Initial Request: Plan a day trip to {city} with interests {interests}\n")
    state = {
        "messages": [HumanMessage(content=f"Plan a day trip to {city} with interests {interests}")],
        "city": city,
        "interests": [i.strip() for i in interests.split(",")], # Ensure interests is a list
        "itinerary": "",
    }

    # ✅ Use invoke instead of stream
    final_state = app.invoke(state)

    if not final_state:
        return "⚠️ Could not generate itinerary."

    return final_state.get("itinerary", "⚠️ No itinerary generated.")

#build the gradio interface
interface = gr.Interface(
    fn = travel_planner,
    theme = 'Yntec/HaleyCH_Theme_Orange_Green',
    inputs=[
        gr.Textbox(label="Enter the city for your day trip: "),
        gr.Textbox(label="Enter your interests (comma-separated)"),
    ],
    outputs=[
        gr.Textbox(label="Generated Itinerary: ",lines=15),
    ],
    title = "🌍 Travel Itinerary Planner",
    description = "Enter a city and your interests to generate a personalised day trip itinerary."
)

interface.launch(share=True)