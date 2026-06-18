RAG-Based Document Retrieval API

Project Overview
This project is a Retrieval-Augmented Generation (RAG) backend application built using FastAPI, Sentence Transformers, LangChain, and ChromaDB. 
It enables semantic search by converting documents into vector embeddings and retrieving the most relevant information based on user queries.

Features
REST API built with FastAPI

Automatic API documentation using Swagger UI

Text chunking using LangChain

Embedding generation using Sentence Transformers

Vector storage and similarity search using ChromaDB

Semantic search for accurate document retrieval

Scalable RAG architecture

Tech Stack

Python
FastAPI
Sentence Transformers
LangChain
ChromaDB

Workflow

Input documents or text.
Split text into smaller chunks using LangChain.
Generate embeddings using Sentence Transformers.
Store embeddings in ChromaDB.
Accept user queries through FastAPI endpoints.
Perform similarity search on stored vectors.
Return the most relevant results.
