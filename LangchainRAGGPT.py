import os
import sys

'''import constants
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.llms import OpenAI
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
#from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
openai = OpenAIEmbeddings(openai_api_key=constants.APIKEY
#from langchain_openai import ChatOpenAI, OpenAIEmbeddings'''


import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain_community.llms import OpenAI
from langchain_community.vectorstores import Chroma

import constants

os.environ["OPENAI_API_KEY"] = constants.APIKEY

#model = OpenAI(model_name="gpt-3.5-turbo-instruct")
model = OpenAIEmbeddings(model="text-embedding-3-large")
query = sys.argv[1]

loader = TextLoader('data/data.txt')
#loader = DirectoryLoader(".", glob="*.txt")
index = VectorstoreIndexCreator().from_loaders([loader])


print(index.query(query, llm = ChatOpenAI()))

