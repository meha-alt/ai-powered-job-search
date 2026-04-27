import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
import pandas as pd
import numpy as np
import json
import requests
from dotenv import load_dotenv
import random
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
API_KEY = os.getenv("API_KEY")
API_ID = os.getenv("API_ID")

client=Groq(api_key=GROQ_API_KEY)

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = load_embeddings()

# Function to extract text from PDF resume
def extract_text(pdf_file):
    reader=PdfReader(pdf_file)
    text=""
    for page in reader.pages:
        text+=page.extract_text()
    return text

# Function to fetch jobs from Rapid API based on role and location
def fetch_jobs(role, location,offset):
    page = offset // 50 + 1  # Adzuna uses page instead of offset

    url = f"https://api.adzuna.com/v1/api/jobs/in/search/{page}"

    params = {
        "app_id": API_ID,
        "app_key": API_KEY,
        "results_per_page": 50,
        "what": role,
        "where": location
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        st.error("API Error")
        return []

    data = response.json()

    jobs = []
    for job in data.get("results", []):
        jobs.append({
            "title": job.get("title", ""),
            "company": job.get("company", {}).get("display_name", ""),
            "description": job.get("description", ""),
            "location": job.get("location", {}).get("display_name", ""),
            "redirect_url": job.get("redirect_url", "")   
        })

    return jobs
# Function to calculate cosine similarity between two vectors
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

#Function to generate personalized email and resume suggestions based on top job recommendations
def generate_suggestions(job,resume):
    response=client.chat.completions.create(
        messages=[ {
        "role": "system",
        "content": "You are a job seeker writing highly personalized, concise, and human-sounding cold emails to professionals for job opportunities. Your goal is to express genuine interest, highlight relevant skills, and politely ask for guidance or opportunities without sounding desperate or generic."
    },
    {
        "role": "user",
        "content": f"""Candidate Resume: {resume}
        Job Details:{job}
        1. Write a personalized cold email:
        - Under 120 words
        - Written from the candidate's perspective
        - Start with a personalized, non-generic opening
        - Mention the role/company specifically
        - Connect candidate’s skills/experience to the job
        - Show curiosity and interest (not desperation)
        - End with a polite, soft call-to-action (e.g., open to a quick chat)
        
        2. Give 3 specific resume improvement suggestions tailored to this job
        
        Format your response EXACTLY like this:
        EMAIL:
        <email text>
        
        RESUME SUGGESTIONS:
        1. ...
        2. ...
        3. ... """}],model="llama-3.3-70b-versatile",temperature=0.5,max_tokens=500)
    return response.choices[0].message.content

# Streamlit UI
st.title("AI-Powered Job Recommendation System")

a=st.text_input("Years of experience")
b=st.text_input("Interested Roles")
c=st.text_input("Skills")
d=st.text_input("Education")
e=st.text_input("Location")
f=st.file_uploader("Upload your resume")
if st.button("Submit"):
    if all([a, b, c, d, e, f]):
        resume_text=extract_text(f)
        st.write("Resume uploaded and processed successfully!")
        jobs = fetch_jobs(b, e,random.randint(0, 50))
        if not jobs:
            jobs = fetch_jobs(b, e, 0)
        scores=[]
        if jobs:
           resume_embedding = embeddings.embed_query(" ".join([a, b, c, d, resume_text[:1000]]))
           for job in jobs:
             job_embedding = embeddings.embed_query(job["description"][:1000])
             score = cosine_similarity(job_embedding, resume_embedding)
             scores.append((score, job))
           scores.sort(reverse=True, key=lambda x: x[0])
           top_jobs = scores[:5]
           st.write("Top Job Recommendations:")
           with st.spinner("Generating personalized emails and resume suggestions..."):
              for score, job in top_jobs:
                  st.write(f"**{job['title']}** at **{job['company']}** in **{job['location']}**")
                  st.write(f"Apply here: {job['redirect_url']}")
                  if(job['description']):
                    st.write(f"Description: {job['description'][:300]}...")
                  st.write(f"Match Score: {score:.4f}")
                  st.write("---")
                  st.write(generate_suggestions(job,resume_text))
                
        else:
          st.write("No job recommendations found.")
    else:
        st.write("Please fill in all fields and upload a resume.")

    