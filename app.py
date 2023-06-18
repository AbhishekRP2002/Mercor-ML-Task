    
import streamlit as st
import requests
import os
from dotenv import load_dotenv
import pandas as pd
from langchain.embeddings.openai import OpenAIEmbeddings
from temp import analysis_main
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
import config




load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

st.set_page_config(page_title="GitHub Repositories List" , page_icon=":computer:" , layout="wide" , initial_sidebar_state="expanded")




def calculate_complexity_score(repo):
    # Implementation of complexity calculation goes here.
    # For this example, I will return a random integer between 1 and 10.
    import random
    return random.randint(1, 10)


# Function to fetch GitHub repositories
@st.cache_data # Cache data so that we don't have to fetch it again
def fetch_github_repos(username):
    url = f'https://api.github.com/users/{username}/repos'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None
    # repos = []
    # page = 1
    # while True:
    #     url = f"https://api.github.com/users/{username}/repos?page={page}&per_page=100"
    #     response = requests.get(url)
    #     data = response.json()
    #     if not data:
    #         break
    #     repos.extend([(repo, calculate_complexity_score(repo)) for repo in data])
    #     page += 1
    # return repos

# Function to display repositories
def display_repos(repos):
    for repo in repos:
        repo_name = repo["name"]
        repo_url = repo["html_url"]
        st.write(f"[{repo_name}]({repo_url})")
 
 
 
def final_analysis(df):
    df.to_csv("data.csv")
    loader = CSVLoader(file_path="data.csv", encoding ="utf-8")
    csv_data = loader.load()
    csv_embeddings = OpenAIEmbeddings()
    vectors = FAISS.from_documents(csv_data, csv_embeddings)
    
    # Create a question-answering chain using the index
    chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=vectors.as_retriever(), input_key="question")
    
    # Pass a query to the chain
    query = """
    You are an inteelligent CSV Agent who can  understand CSV files and their contents. You are given a CSV file with the following columns: Repository Name, Repository Link, Analysis. You are asked to find the most technically complex and challenging repository from the given CSV file. 
    
    What is the most technically challenging repository from the given CSV file?
    Return the name of the repository , the link to the repository and the analysis of the repository showing why it is the most technically challenging/Complex repository.
    
    The output should be in the following format:
    
    Repository Name: <name of the repository>
    Repository Link: <link to the repository>
    Analysis: <analysis of the repository>
    
    Provide a clickable link to the repository as well like this:
    [Repository Name](Repository Link)
    
    """
    response = chain({"question": query})
    st.write(response['result'])
    st.stop()

    

# Main app
def main():
    config.init()
    # Set up the app title and sidebar
    st.title("GitHub Automated Analysis Tool")
    st.sidebar.title("GitHub Automated Analysis Tool")

    # Input field for GitHub username
    username = st.sidebar.text_input("Enter GitHub Username")

    # Submit and clear buttons
    submit_button = st.sidebar.button("Submit")
    clear_button = st.sidebar.button("Clear")
    st.sidebar.header("About")
    st.sidebar.info("This Python-based tool , when given a GitHub user's URL, returns the most technically complex and challenging repository from that user's profile. The tool will use GPT and LangChain to assess each repository individually before determining the most technically challenging one.")
    st.divider()
    st.sidebar.write("This tool is created by  [Abhishek Ranjan](https/github.com/AbhishekRP2002).")

    # Display the repositories
    if submit_button:
        st.subheader(f"Repositories for {username}")
        repos = fetch_github_repos(username)
        if repos:
            display_repos(repos)
            st.info("Analysis of the repositories using LangChain and ChatGPT started. Please wait...")
            for repo_detail in repos:
                repository_name = repo_detail['name']
                repository_url = repo_detail['html_url']
                name , link , analysis = analysis_main(repository_name, repository_url)
                config.main_df =  pd.concat([config.main_df, pd.DataFrame([{'Repository Name': str(name) , 
                                                                            'Repository Link':str(link) ,
                                                                            'Analysis': str(analysis)
                                                                            }])], ignore_index=True)
            st.dataframe(config.main_df)
            final_analysis(config.main_df)
        else:
            st.error("Invalid username or unable to fetch repositories")

    # Clear the input field
    if clear_button:
        username = ""
        st.experimental_rerun()





if __name__ == "__main__":
    main()