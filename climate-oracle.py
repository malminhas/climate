#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# climate-oracle.py
# -----------------
# (c) 2024 Mal Minhas, <mal@malm.co.uk>
#
# RAG climate question and answer CLI built on IPCC AR6 Synthesis Report.
#
# Installation:
# -------------
# pip install -r requirements.txt
#
# Implementation:
# --------------
# CLI leverages the code built in the accompanying notebook.
#
# TODO:
# -----
#
# History:
# -------
# 26.11.24    v0.1    First cut based on accompanying notebook
# 26.11.24    v0.2    Added Anthropic and Ollama LLMs plus CLI support
# 29.11.24    v0.3    First drop into climate repository
# 07.12.24    v0.4    Added cost output and force to regenerate vectorstore
# 08.12.24    v0.5    Added tokencost for pricing, references switched to Ollama default
# 30.12.24    v0.6    Added groq support

import os
import time
import logging
from logging import Logger
from typing import List, Dict, Tuple
import requests
from requests.exceptions import ConnectionError

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI, OpenAI
from langchain_ollama.llms import OllamaLLM
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.callbacks import get_openai_callback
from langchain_core.runnables import Runnable
from langchain_core.language_models import LLM

PROGRAM = __file__.split("/")[-1]
VERSION = "0.6"
DATE = "30.12.24"
AUTHOR = "<mal@malm.co.uk>"

VALID_MODELS = ['ollama', 'gpt-3.5', 'gpt-4', 'claude', 'groq']
IPCC_AR6_PDF = "https://www.ipcc.ch/report/ar6/syr/downloads/report/IPCC_AR6_SYR_LongerReport.pdf"
OLLAMA_URL = "http://localhost:11434"

system_prompt = """You are the Climate Assistant, a helpful AI assistant.
Your task is to answer common questions on climate change.
You will be given a question and relevant excerpts from the IPCC Climate Change Synthesis Report (2023).
Please provide comprehensive answers based on the provided context.  Be polite and helpful.

Context:
{context}

Question:
{question}

Your answer:
"""

def initLogger(verbose: bool) -> Logger:
    ''' Initialise standard Python console logging. '''
    if verbose:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s :: %(levelname)s :: %(message)s"
        )
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        # Suppress pypdf 'Ignoring wrong pointing object' warnings
        pypdf_logger = logging.getLogger("pypdf")
        pypdf_logger.setLevel(logging.WARNING)
    else:
        # this will silence all logging including from modules
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.FileHandler(os.devnull))
        # Suppress pypdf 'Ignoring wrong pointing object' warnings
        pypdf_logger = logging.getLogger("pypdf")
        pypdf_logger.setLevel(logging.ERROR)
    return logger

def is_ollama_running(logger: Logger) -> bool:
    ''' Check if Ollama is running locally '''
    logger.info(f"::is_ollama_running()")
    try:
        response = requests.get(f'{OLLAMA_URL}/api/tags')
        return response.status_code == 200
    except ConnectionError as e:
        logger.error(f"Ollama is not running: {e}")
        return False

def getModel(model_provider: str, logger: Logger) -> LLM:
    ''' Create and return LLM instance based on input string. '''
    logger.info(f"::getModel({model_provider})")
    model_name = ''
    if model_provider in ['gpt-3.5']:
        # See: https://python.langchain.com/docs/integrations/providers/openai
        # pip install -U langchain-openai
        model_name = 'gpt-3.5-turbo-instruct'
        llm = OpenAI(model=model_name, temperature=0)
    elif model_provider in ['gpt-4']:
        # See: https://python.langchain.com/docs/integrations/providers/o1
        # pip install -U langchain-o1
        model_name = 'gpt-4o'
        llm = ChatOpenAI(model=model_name, temperature=0)
    elif model_provider in ['claude']:
        # See: https://python.langchain.com/docs/integrations/providers/anthropic/
        # pip install -U langchain-anthropic
        # Model to use is 'claude-3-5-sonnet-latest'
        #name = 'claude-3-5-sonnet-latest'
        #llm = ChatAnthropic(model=name, temperature=0)
        model_name = 'claude-3-5-sonnet-latest'
        print(f"Using Anthropic model='{model_name}'")
        llm = ChatAnthropic(model_name=model_name, temperature=0)
    elif model_provider in ['groq']:
        # See: https://python.langchain.com/docs/integrations/providers/groq/
        # pip install -U langchain-groq
        model_name = 'mixtral-8x7b-32768' # limited to 5000 characters
        print(f"Using Groq model='{model_name}'")
        llm = ChatGroq(model_name=model_name, temperature=0)
    elif model_provider in ['ollama']:
        # Default to Ollama if no model name passed in at CLI - check if it is running locally first
        logger.info("Checking if Ollama is running...")
        if not is_ollama_running(logger):
            raise RuntimeError(f"Ollama is not found at {OLLAMA_URL}. It needs to be running first.")
        # See: https://python.langchain.com/docs/integrations/providers/ollama/
        # pip install -U langchain-ollama
        model_name = 'llama3.2'
        llm = OllamaLLM(model=model_name, temperature=0)
    else:
        raise ValueError(f"Invalid model provider: '{model_provider}'. Valid options are: {VALID_MODELS}")
    
    return model_name, llm

def generateChain(pdfs: List, model_provider: str, logger: Logger, force: bool = False) -> Tuple[str, Runnable]:
    ''' Create RAG chain. '''
    logger.info(f"::generateChain({model_provider})")
    index_file = 'faiss_index'
    if not force and os.path.exists(index_file):
        print(f"Loading vectorstore from '{index_file}'")
        vectorstore = FAISS.load_local(index_file, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    else:
        print(f"Preparing and saving vectorstore to '{index_file}'")
        # Prepare vector store (FAISS) with IPPC report(s).  Store splits in vectorstore
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
        for i, pdf in enumerate(pdfs):
            loader = PyPDFLoader(pdf)
            if i == 0:
                vectorstore = FAISS.from_documents(documents=loader.load_and_split(text_splitter), embedding=OpenAIEmbeddings())
            else:
                vectorstore_i = FAISS.from_documents(documents=loader.load_and_split(text_splitter), embedding=OpenAIEmbeddings())
                vectorstore.merge_from(vectorstore_i)
        vectorstore.save_local(index_file)
        
    # Original recipe had number of chunks as len(pdfs) but this is too few for a single document
    #retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": len(pdfs)})
    model_name, llm = getModel(model_provider, logger)
    if model_provider == 'groq':
        print(f"For groq we are setting k to 5")
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    else:
        print(f"For {model_provider} we are setting k to 30")
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 30})
    # Question-answering against an index using create_retrieval_chain:
    prompt = PromptTemplate(template=system_prompt, input_variables=["question", "context"])
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    qa_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=document_chain)
    return model_name, qa_chain

def getAnswer(model_name: str, qa_chain: Runnable, question: str, logger: Logger) -> Dict:
    ''' Get answer. '''
    logger.info(f"::getAnswer({model_name}, Q:'{question}')")
    r = {}

    # 1. Get the answer from the RAG results
    answer = qa_chain.invoke({"input": system_prompt,"question": question})
    r['answer'] = answer.get('answer').strip()
    answer_text = r['answer']

    # 2. Get the full context from the RAG results
    context = answer.get('context', [])
    context_text = "\n".join([doc.page_content for doc in context]) if context else ""
    # Construct the full prompt including RAG content
    full_prompt = system_prompt.format(context=context_text, question=question)
    logger.info(f"Full prompt with RAG content:\n{full_prompt}")
    
    # Add RAG references to return object
    calculateReferences(answer, r, logger)

    # 3. Get price of running the query
    # Prices obtained from looking at pricing pages for each model per million tokens in USD:
    # https://www.anthropic.com/pricing#anthropic-api
    # https://platform.openai.com/pricing
    # Derive counts using full context prompt
    r['prompt_tokens'] = len(full_prompt.split())
    r['completion_tokens'] = len(answer_text.split())
    r['total_tokens'] = r['prompt_tokens'] + r['completion_tokens']
    calculateCosts(model_name, r, logger)
    
    return r

def calculateReferences(answer: Dict, r: Dict, logger: Logger) -> None:
    ''' Calculate references. '''
    logger.info(f"::calculateReferences()")
    r['references'] = []
    r['context'] = []
    for i,doc in enumerate(answer.get('context')):
        ref = doc.to_json()
        source = f"source: {ref.get('kwargs').get('metadata').get('source')}"
        page = f"page: {ref.get('kwargs').get('metadata').get('page')}"
        chunk = ref.get('kwargs').get('page_content')
        chunklen = len(chunk)
        contents = f'chunk length: {chunklen}, snippet: "{chunk}"'
        r['references'].append(f"{source}, {page}")
        r['context'].append(contents)

def calculateCosts(model_name: str, r: Dict, logger: Logger) -> None:
    ''' Calculate costs. '''
    logger.info(f"::calculateCosts({model_name})")
    mapping = {'llama3.2': [0.0, 0.0],
               'gpt-3.5-turbo-instruct': [1.5, 2.00],
               'mixtral-8x7b-32768': [0.1, 0.1],
               'gpt-4o': [2.5, 10.00],
               'claude-3-5-sonnet-latest': [3.0, 15.00]}
    pc, cc = mapping[model_name] or [0.0, 0.0]
    prompt_cost = r.get('prompt_tokens') * pc/1000000
    completion_cost = r.get('completion_tokens') * cc/1000000
    r['cost'] = prompt_cost + completion_cost

def formatCosts(answer: Dict, logger: Logger) -> str:
    ''' Format costs. '''
    logger.info(f"::formatCosts()")
    dollar_cost = answer.get('cost')
    ptokens = answer.get('prompt_tokens')
    ctokens = answer.get('completion_tokens')
    ttokens = answer.get('total_tokens')
    amount = round(dollar_cost, 4)
    s = "---------- COST ----------\n"
    s += f'cost=${amount}, prompt_tokens={ptokens}, completion_tokens={ctokens}, total_tokens={ttokens}'
    return s

def formatReferences(answer: Dict, logger: Logger) -> str:
    ''' Format references. '''
    snippet_length = 132
    s = "\n---------- REFERENCES ----------\n"
    for i,reference in enumerate(answer.get('references')):
        chunk = answer.get('context')[i]
        s += f" ======== RAG REFERENCE {i+1} ========\n{reference}\n{chunk[:snippet_length]}"
        if len(chunk) > snippet_length:
            s += "...\n"
    return s

def main(arguments: Dict):
    verbose = False
    cost = False
    force = False
    references = False
    logger = initLogger(False)
    model_provider = "ollama"
    if arguments.get("--verbose"):
        verbose = True
        logger = initLogger(verbose)
        logger.info(f"::main() - arguments =\n{arguments}")
    if arguments.get("--model"):
        model_provider = str(arguments.get("--model")[0])
    if arguments.get("--cost"):
        cost = True
    if arguments.get("--force"):
        force = True
    if arguments.get("--references"):
        references = True

    if arguments.get("--version") or arguments.get("-V"):
        print(f"{PROGRAM} version {VERSION}.  Released {DATE} by {AUTHOR}")
    elif arguments.get("--help") or arguments.get("-h"):
        print(usage)
    else:
        t0 = time.time()
        pdfs = [IPCC_AR6_PDF]
        print(f"Attempting to create_retrieval_chain using llm='{model_provider}'...")
        try:
            model_name, climate_qa_chain = generateChain(pdfs, model_provider, logger, force)
        except Exception as e:
            print(f"Error generating retrieval chain: {e}\nPlease ensure dependencies are installed and started.")
            return
        t1 = time.time()
        print(f"successfully loaded RAG in {round((t1-t0),2)} seconds and will use '{model_name}' LLM")
        print("Enter a climate question (or press Enter to quit). eg. 'Is sea level rise avoidable and when will it stop?'")
        while True:
            try:
                # Prompt the user to enter a sentence
                question = input("> ")
                # Break the loop if the user enters an empty string
                if question == "":
                    print("Empty input.  Exiting the program. Goodbye!")
                    break
                answer = getAnswer(model_name, climate_qa_chain, question, logger)
                print(f"{answer.get('answer')}")
                s = ''
                if cost:
                    s += formatCosts(answer, logger)
                if references and answer.get('references'):
                    s += formatReferences(answer, logger)
                print(s)
            except Exception as e:
                print(f"Exception: {e}.  Exiting the program. Goodbye!")
                break

if __name__ == "__main__":
    import docopt

    usage = """
    {}
    ------------------
    Usage:
    {} [-m <model>] [-c] [-r] [-f] [-v]
    {} -h | --help
    {} -V | --version
    Options:
    -h, --help                          Show this screen.
    -v, --verbose                       Verbose mode.
    -V, --version                       Show version.
    -m <model>, --model <model>         LLM. Default is ollama. Options: gpt-3.5, gpt-4, claude, groq.
    -c, --cost                          Return cost of call in dollars + tokens
    -r, --references                    Return RAG references
    -f, --force                         Force regeneration of vectorstore.
    Examples:
    1. Create verbose climate oracle using Anthropic claude-sonnet LLM:
    {} -m claude -v
    2. Create climate oracle with cost output using OpenAI gpt-3.5-turbo-instruct LLM:
    {} -m gpt-3.5 -c
    3. Create climate oracle using default Ollama and force regeneration of vectorstore:
    {} -c -f
    4. Create climate oracle with references using gpt-4o LLM:
    {} -m gpt-4 -c -r
    """.format(
        *tuple([PROGRAM] * 8)
    )
    arguments = docopt.docopt(usage)
    main(arguments)
