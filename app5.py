import streamlit as st
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler

import os

from dotenv import load_dotenv
load_dotenv()

# Streamlit
st.title("Langchain - Chat with Search")

#Arxiv tool
api_wrapper_wiki=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

#wikipedia tool
api_wrapper_arxiv=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=250)
arxiv=ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

# DuckDuckGoSearch tool
search=DuckDuckGoSearchRun(name="Search")

#Input the groq api key
api_key=st.text_input("Enter the Groq API key :",type="password")

#Initialize the chat history
if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant",
        "content":"Hi,I'm a chatbot who can search the web.How can i help you?"}
    ]

#Display chat message from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

#React to the user input
if prompt:=st.chat_input(placeholder="Say something"):
    #Display user message in chat message container
    with st.chat_message("user"):
        st.write(prompt)
    #Add user message to the chat history
    st.session_state.messages.append({"role":"user","content":prompt})
    

    tools=[search,arxiv,wiki]
    llm=ChatGroq(model_name="Gemma2-9b-It",groq_api_key=api_key,streaming=True)
    search_agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_error=True)

    #Display assistant response in chat message container
    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response=search_agent.run(st.session_state.messages,callbacks=[st_cb])
        #Add assistant response in chat history
        st.session_state.messages.append({"role":"assistant","content":response})
        st.write(response)
