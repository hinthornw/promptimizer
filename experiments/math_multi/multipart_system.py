from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


class Prompts(TypedDict):
    classifier: ChatPromptTemplate
    math_solver: ChatPromptTemplate


llm = ChatOpenAI(model="gpt-4o-mini")


async def solve_problem(prompts: Prompts, inputs: dict) -> str:
    result_message = await llm.ainvoke(prompts["classifier"].invoke(inputs))
    language = result_message.content
    response = await llm.ainvoke(
        prompts["math_solver"].invoke({"language": language, **inputs})
    )
    return response.content
