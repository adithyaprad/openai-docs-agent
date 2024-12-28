

import os
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, WebsiteSearchTool

openai_api_key = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_MODEL_NAME"] = 'gpt-4o-mini'

inquiry = input("Enter inquiry: ").strip()

support_agent = Agent(
    role="Senior Support Representative",
    goal="Be the most friendly and helpful support representative in your team",
    backstory=(
        "You work at OpenAI (https://platform.openai.com/docs) and are now working on providing "
        "support to {customer}, a super important customer for your company. "
        "You need to make sure that you provide the best support! "
        "You have access to all the docs on the OpenAI platform. "
        "Make sure to provide full, complete answers, and make no assumptions. "
        "When including code snippets, provide the entire code together without breaking it apart, "
        "so it's easy to copy and paste if needed. "
        "Ensure that code snippets are included in the final response when available in the specific website docs."
    ),
    verbose=True
)

support_quality_assurance_agent = Agent(
    role="Support Quality Assurance Specialist",
    goal="Get recognition for providing the best support quality assurance in your team",
    backstory=(
        "You work at OpenAI (https://platform.openai.com/docs) and are now working with your team "
        "on a request from {customer} ensuring that the support representative is providing the best support possible. "
        "You need to make sure that the support representative is providing full, complete answers, meeting all the user requirements, and make no assumptions. "
        "Ensure that code snippets are provided in full without breaking them apart."
    ),
    allow_code_execution=True,
    verbose=True
)

final_review_agent = Agent(
    role="Final Review Specialist",
    goal="Ensure the final response is clear, concise, and free from repetition or confusion.",
    backstory=(
        "You are an expert in content editing and quality control. "
        "Your role is to review the final customer support response and ensure it is polished, concise, and completely free of redundancies or mixed-up information. "
        "Ensure that code snippets are provided in full without breaking them apart."
    ),
    allow_code_execution=True,
    verbose=True
)

docs_scrape_tool = ScrapeWebsiteTool()
search_tool = SerperDevTool()

inquiry_resolution = Task(
    description=(
        "{customer} just reached out with a super important ask:\n"
        "{inquiry}\n\n"
        "{person} from {customer} is the one that reached out. "
        "Make sure to use everything you know to provide the best support possible. "
        "Lookup on Google the OpenAI docs (website - https://platform.openai.com/docs) related to the user inquiry and use the information found on the documentation website. "
        "You are only to stick to the information found on the documentation website and not deviate to information found on other websites. "
        "Make sure that when you are accessing information from the website, you at all times have to access the code snippets of the information. "
        "The user has a specific request that code snippets related to the query have to be included in the final response; this is only if they are available in the specific website docs. "
        
        "The query might contain multiple questions that might need you to go to different websites that start with https://platform.openai.com/docs/. "
        "You must do so to answer every question that the customer asks you. "
        "You must strive to provide a complete and accurate response to the customer's inquiry. "
        "When including code snippets, provide the entire code together without breaking it apart, so it's easy to copy and paste if needed."
    ),
    expected_output=(
        "A detailed, informative response to the customer's inquiry that addresses all aspects of their question. "
        "The response should include references to everything you used to find the answer, including external data or solutions. "
        "Ensure the answer is complete, leaving no questions unanswered, and maintain a helpful and friendly tone throughout. "
        "When including code snippets, provide them in full without breaking them apart."
    ),
    tools=[docs_scrape_tool, search_tool],
    agent=support_agent
)

quality_assurance_review = Task(
    description=(
        "Review the response drafted by the Senior Support Representative for {customer}'s inquiry. "
        "Ensure that the answer is comprehensive, accurate, and adheres to the customer's inquiry. "
        "Ensure that every part of the question is answered, even if it takes the first agent looking up multiple websites to answer the question. "
        "Ensure that all the user's demands are met, mainly the inclusion of code snippets in the final response (only if they are part of the website docs). "
        "High-quality standards are expected for customer support. "
        "Verify that all parts of the customer's inquiry have been addressed thoroughly, with a helpful and friendly tone. "
        "Check for references and sources used to find the information, ensuring the response is well-supported and leaves no questions unanswered. "
        "When including code snippets, ensure they are provided in full without breaking them apart."
        "you should also make sure that the code snippet that is provided is correct . it might be outdated , or syntactically wrong. you need to make sure it is"
    ),
    expected_output=(
        "A final, detailed, and informative response ready to be sent to the customer. "
        "This response should fully address the customer's inquiry, incorporating all relevant feedback and improvements. "
        "Don't be too formal; we are a chill and cool company but maintain a professional and friendly tone throughout. "
        "When including code snippets, provide them in full without breaking them apart."
    ),
    agent=support_quality_assurance_agent, 
    tools=[docs_scrape_tool, search_tool],

    allow_delegation=True
)

final_review_task = Task(
    description=(
        "Review the response produced by the Support and QA agents. "
        "Your goal is to eliminate any repetition, confusion, or verbosity, ensuring the response is concise, clear, and professional. "
        "Maintain a friendly and helpful tone throughout. "
        "Ensure that code snippets are provided in full without breaking them apart."

    ),
    expected_output=(
        "A refined and polished response that is free from repetition, confusion, and unnecessary verbosity, ready to be sent to the customer. "
        "Code snippets should be provided in full without breaking them apart."
    ),
    agent=final_review_agent

)

crew = Crew(
    agents=[support_agent, support_quality_assurance_agent, final_review_agent],
    tasks=[inquiry_resolution, quality_assurance_review, final_review_task],
    verbose=True,
    memory=False
)

inputs = {
    "customer": "DeepLearningAI",
    "person": "Adi",
    "inquiry": inquiry
}

result = crew.kickoff(inputs=inputs)

print(result)
