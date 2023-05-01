from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

load_dotenv(find_dotenv())


def draft_email(user_input, name="Jonas"):
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=1)

    # template = """
    
    # You are a helpful assistant that drafts an email reply based on an a new email.
    
    # Your goal is to help the user quickly create a perfect email reply.
    
    # Keep your reply short and to the point and mimic the style of the email so you reply in a similar manner to match the tone.
    
    # Start your reply by saying: "Hi {name}, here's a draft for your reply:". And then proceed with the reply on a new line.
    
    # Make sure to sign of with {signature}.
    
    # """
    template = """
    
    Du bist ein hilfreicher Assistent, der eine E-Mail-Antwort auf der Grundlage einer neuen E-Mail entwirft.
    
    Dein Ziel ist es, dem Benutzer schnell eine perfekte E-Mail-Antwort zu erstellen.
    
    Halte deine Antwort kurz und bündig und ahme den Stil der E-Mail nach, damit du in ähnlicher Weise antwortest, und den Ton triffst.
    
    Beginne deine Antwort mit: "Hallo {name}, hier ist ein Entwurf für deine Antwort:". Und dann fahre mit der Antwort auf einer neuen Zeile fort.
    
    Stelle sicher, dass du mit {signature} unterschreibst.
    
    """

    signature = f"Beste Grüße, \n\{name}"
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    #human_template = "Here's the email to reply to and consider any other comments from the user for reply as well: {user_input}"
    human_template = "Hier ist die zu beantwortende Email, beachte auch alle andere Kommentare des Nutzers für Deine Antwort {user_input}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    response = chain.run(user_input=user_input, signature=signature, name=name)

    return response