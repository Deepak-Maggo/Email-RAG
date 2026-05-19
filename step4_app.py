# python -m streamlit run step4_app.py
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader

# ==========================================
# 1. SETUP THE WEB PAGE
# ==========================================
st.set_page_config(page_title="AI Agent", page_icon="📧")
st.title("📧 Custom Email Agent")
st.write("Type your query, and the AI will write the perfect email based on your standard templates.")

# ==========================================
# 2. LOAD THE AI (CACHED)
# ==========================================
# @st.cache_resource is a magic Streamlit command. 
# It keeps the AI loaded in the background so it doesn't have to 
# reboot the database every time you click "Generate".
@st.cache_resource
def load_brain_and_memory():
    load_dotenv()
    # Wake up Groq
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))
    # Wake up Local Database
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if not os.path.exists("./my_local_db"):
        print("No database found! Building it now...")
        all_documents = []
        # Loop through the data folder and load all templates
        for filename in os.listdir("./data"):
            if filename.endswith(".txt"):
                # Make sure to use utf-8 encoding just like we fixed earlier!
                loader = TextLoader(f"./data/{filename}", encoding="utf-8")
                all_documents.extend(loader.load())
        
        # Build and save the database
        db = Chroma.from_documents(
            documents=all_documents, 
            embedding=embedding_model, 
            persist_directory="./my_local_db" 
        )
    else:
        # If it already exists, just load it normally
        # db = Chroma(persist_directory="./my_local_db", embedding_function=embedding_model)
        db = Chroma(persist_directory="./my_local_db", embedding_function=embedding_model)
    return llm, db

llm, database = load_brain_and_memory()

# ==========================================
# 3. THE USER INTERFACE
# ==========================================
# Create a big text box for you to type in
user_request = st.text_area("What do you need an email for today?", height=150, placeholder="e.g., This setup is not compliant under the section 12 of companies act")

# Create a shiny Generate button
if st.button("Generate Email"):
    
    if user_request: # If the text box isn't empty
        with st.spinner("Searching rulebook and writing email..."):
            
            # 1. Search the Database
            search_results = database.similarity_search(user_request, k=1)
            found_template = search_results[0].page_content
            
            # 2. The Hybrid Prompt
            master_prompt = f"""
            You are an expert email assistant for Deepak Maggo.

            GLOBAL FORMATTING RULES:
            Every single email you write MUST start exactly with:
            "Dear Sir/Mam,\nHope this email finds you well,"

            Every single email you write MUST end exactly with:
            "Best regards,\nDeepak Maggo"

            KNOWLEDGE BASE (Reference Data):
            {found_template}

            USER REQUEST:
            {user_request}

            INSTRUCTIONS:
            1. Draft a professional email based on the USER REQUEST.
            2. facts and arguments from the KNOWLEDGE BASE can be used to answer if found relevant do not blindly trust those templates.
            3. CRITICAL: Do not just blindly copy-paste. Weave the information naturally into the email.
            """
            
            # 3. Get AI Response
            response = llm.invoke(master_prompt)
            
            # 4. Display the result on the web page!
            st.success("Email Generated!")
            st.write("---")
            st.write(response.content)
            
    else:
        st.warning("Please type a request into the box first!")