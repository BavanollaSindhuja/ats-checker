import streamlit as st
import time
import re
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import PyPDF2
import docx2txt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import google.generativeai as genai
import os
from dotenv import load_dotenv
from google.api_core import exceptions

# Load environment variables from .env file
load_dotenv()

# Download necessary NLTK data at the beginning of the script
nltk.download('punkt')
nltk.download('stopwords')

# Configure Google Gemini API
# Configure Google Gemini API dynamically with preferred model selection
# Configure Google Gemini API dynamically with preferred model selection
try:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)

        preferred_models = [
            "gemini-1.5-flash",  # Use SHORT names without "models/" prefix
            "gemini-1.5-pro",
            "gemini-pro"
        ]

        # Get available models
        available_models = [model.name for model in genai.list_models()]
        
        # Find first available preferred model
        gemini_model = next((m for m in preferred_models if m in available_models), None)

        if gemini_model:
            # Create GenerativeModel instance, not Model
            model = genai.GenerativeModel(gemini_model)  # FIXED: Use GenerativeModel
            gemini_available = True
        else:
            gemini_available = False
    else:
        gemini_available = False
except Exception as e:
    print(f"Error configuring Gemini: {e}")
    gemini_available = False

def extract_text_from_resume(uploaded_file):
    if uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return docx2txt.process(uploaded_file)
    elif uploaded_file.type == "text/plain":
        return str(uploaded_file.read(), "utf-8")
    else:
        return None

import re

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    stopwords_list = {
        'a', 'an', 'the', 'and', 'or', 'if', 'to', 'of', 'in', 'on', 'for', 'with', 
        'at', 'by', 'from', 'up', 'about', 'as', 'into', 'like', 'through', 'after',
        'over', 'between', 'out', 'against', 'during', 'without', 'before', 'under',
        'around', 'among'
    }
    return [word for word in tokens if word not in stopwords_list]


def extract_keywords(text):
    tokens = preprocess_text(text)
  
    freq_dist = nltk.FreqDist(tokens)
    
    keywords = [word for word, freq in freq_dist.most_common(30) if len(word) > 3]
    return keywords


def calculate_ats_score(resume_text, job_description):
    if not resume_text or not job_description:
        return 0, [], []
    
    
    job_keywords = extract_keywords(job_description)
    
    
    resume_tokens = set(preprocess_text(resume_text))
    
    
    matching_keywords = [keyword for keyword in job_keywords if keyword in resume_tokens]
    missing_keywords = [keyword for keyword in job_keywords if keyword not in resume_tokens]
    
    
    score = len(matching_keywords) / len(job_keywords) * 100 if job_keywords else 0
    
    return score, matching_keywords, missing_keywords


# In get_gemini_recommendations() (replace this):
def get_gemini_recommendations(resume_text, job_description, matching_keywords, missing_keywords, score):
    if not gemini_available:
        return "To enable AI-powered recommendations, please add your Gemini API key in the settings."

    try:
        # Proper model name cleanup
        preferred_models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
        available_models = [model.name.split("/")[-1] for model in genai.list_models()]
        selected_model = next((m for m in preferred_models if m in available_models), None)

        if not selected_model:
            return "No compatible Gemini models available"

        # Use fully-qualified model name
        full_model_name = f"models/{selected_model}"
        model = genai.GenerativeModel(full_model_name)

        # Define the prompt content
        prompt = f"""
You are an expert career coach and ATS optimization specialist.

Analyze the following resume and job description. Based on the given ATS score and keyword analysis, provide specific, actionable improvements that can help increase the resume's alignment with the job description.

Resume:
\"\"\"{resume_text}\"\"\"

Job Description:
\"\"\"{job_description}\"\"\"

Matching Keywords:
{', '.join(matching_keywords)}

Missing Keywords:
{', '.join(missing_keywords)}

Current ATS Match Score: {score:.1f}%

Instructions:
- List 3 to 5 detailed recommendations for enhancing the resume.
- Focus on incorporating missing keywords naturally.
- Suggest improvements in wording, formatting, and alignment to job expectations.
- Use clear, concise bullet points.
"""

        # Generate response
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"Error generating AI recommendations: {str(e)}"
    
# Function to display score meter
def display_score_meter(score):
    fig, ax = plt.subplots(figsize=(10, 2))
    
    # Create a horizontal bar
    ax.barh([0], [100], color='lightgray', height=0.5)
    ax.barh([0], [score], color=get_color_based_on_score(score), height=0.5)
    
    # Add score text in the middle
    ax.text(50, 0, f"{score:.1f}%", 
            ha='center', va='center', 
            fontsize=14, fontweight='bold', 
            color='black')
    
    # Set limits and remove ticks
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([])
    
    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Add labels for the score ranges
    ax.text(12.5, -0.3, "Poor", ha='center', va='top', color='darkred')
    ax.text(37.5, -0.3, "Average", ha='center', va='top', color='darkorange')
    ax.text(62.5, -0.3, "Good", ha='center', va='top', color='darkgreen')
    ax.text(87.5, -0.3, "Excellent", ha='center', va='top', color='darkblue')
    
    plt.tight_layout()
    return fig

# Function to get color based on score
def get_color_based_on_score(score):
    if score < 25:
        return 'darkred'
    elif score < 50:
        return 'darkorange'
    elif score < 75:
        return 'yellowgreen'
    else:
        return 'darkgreen'

# Function to generate basic recommendations
def generate_recommendations(score, matching_keywords, missing_keywords):
    recommendations = []
    
    if score < 25:
        recommendations.append("Your resume needs significant improvement to pass ATS systems.")
    elif score < 50:
        recommendations.append("Your resume needs some improvement to better match this job description.")
    elif score < 75:
        recommendations.append("Your resume is doing well but could be optimized further.")
    else:
        recommendations.append("Your resume is well-optimized for this job description!")
    
    # Add specific recommendations
    if missing_keywords:
        recommendations.append(f"Consider adding these missing keywords: {', '.join(missing_keywords[:10])}" + 
                              (f" and {len(missing_keywords) - 10} more..." if len(missing_keywords) > 10 else ""))
    
    return recommendations

# Sidebar for API key configuration
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Enter Gemini API Key", type="password", help="Get your API key from https://ai.google.dev/")
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
        genai.configure(api_key=api_key)
        gemini_available = True
        st.success("API key configured successfully!")
        
        # Show available models when API key is configured
        try:
            models = genai.list_models()
            st.write("Available models:")
            for model in models:
                st.write(f"- {model.name}")
        except Exception as e:
            st.error(f"Error listing models: {str(e)}")
    
    st.write("---")
    st.write("About")
    st.write("This ATS Score Checker helps you optimize your resume for Applicant Tracking Systems by analyzing keyword matches and providing AI-powered recommendations.")

# Main application UI
st.title("ATS Resume Score Checker")
st.markdown("Upload your resume and paste a job description to see how well your resume matches the job requirements.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Resume")
    uploaded_file = st.file_uploader("Choose your resume file", type=["pdf", "docx", "txt"])
    
    if uploaded_file is not None:
        st.session_state.resume_text = extract_text_from_resume(uploaded_file)
        st.success("Resume uploaded successfully!")
        
        with st.expander("View Resume Text"):
            st.text_area("Extracted Text", st.session_state.resume_text, height=300)
    
    st.subheader("Or Paste Resume Text")
    resume_text_input = st.text_area("Paste your resume text here", height=200)
    if resume_text_input:
        st.session_state.resume_text = resume_text_input

with col2:
    st.subheader("Job Description")
    job_description = st.text_area("Paste the job description here", height=300)
    if job_description:
        st.session_state.job_description = job_description

# Add a check button
if st.button("Check ATS Score"):
    if not st.session_state.resume_text:
        st.error("Please upload or paste your resume text.")
    elif not st.session_state.job_description:
        st.error("Please paste the job description.")
    else:
        with st.spinner("Analyzing your resume..."):
            # Calculate ATS score
            score, matching_keywords, missing_keywords = calculate_ats_score(
                st.session_state.resume_text, 
                st.session_state.job_description
            )
            
            st.session_state.score = score
            st.session_state.matching_keywords = matching_keywords
            st.session_state.missing_keywords = missing_keywords
            
            # Get AI recommendations if Gemini API is available
            if gemini_available:
                with st.spinner("Generating AI recommendations..."):
                    st.session_state.ai_recommendations = get_gemini_recommendations(
                        st.session_state.resume_text,
                        st.session_state.job_description,
                        matching_keywords,
                        missing_keywords,
                        score
                    )

# Display results if available
if hasattr(st.session_state, 'score') and st.session_state.score > 0:
    st.header("ATS Score Results")
    
    # Display score meter
    st.subheader(f"Your ATS Score: {st.session_state.score:.1f}%")
    score_meter = display_score_meter(st.session_state.score)
    st.pyplot(score_meter)
    
    # Create tabs for different sections of the analysis
    tab1, tab2, tab3 = st.tabs(["Keywords Analysis", "Basic Recommendations", "AI Analysis"])
    
    with tab1:
        # Display matching and missing keywords
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Matching Keywords")
            if st.session_state.matching_keywords:
                for keyword in st.session_state.matching_keywords:
                    st.markdown(f"✅ {keyword}")
            else:
                st.write("No matching keywords found!")
        
        with col2:
            st.subheader("Missing Keywords")
            if st.session_state.missing_keywords:
                for keyword in st.session_state.missing_keywords[:10]:
                    st.markdown(f"❌ {keyword}")
                if len(st.session_state.missing_keywords) > 10:
                    st.write(f"... and {len(st.session_state.missing_keywords) - 10} more")
            else:
                st.write("No missing keywords found!")
    
    with tab2:
        # Display basic recommendations
        st.subheader("Basic Recommendations")
        recommendations = generate_recommendations(
            st.session_state.score, 
            st.session_state.matching_keywords, 
            st.session_state.missing_keywords
        )
        for rec in recommendations:
            st.markdown(f"- {rec}")
    
    with tab3:
        # Display AI-powered recommendations
        st.subheader("AI-Powered Analysis")
        if gemini_available and st.session_state.ai_recommendations:
            st.markdown(st.session_state.ai_recommendations)
        elif not gemini_available:
            st.info("To enable AI-powered recommendations, please add your Gemini API key in the sidebar settings.")
            if st.button("Get AI Recommendations"):
                if not api_key:
                    st.error("Please enter your Gemini API key in the sidebar first.")
                else:
                    with st.spinner("Generating AI recommendations..."):
                        st.session_state.ai_recommendations = get_gemini_recommendations(
                            st.session_state.resume_text,
                            st.session_state.job_description,
                            st.session_state.matching_keywords,
                            st.session_state.missing_keywords,
                            st.session_state.score
                        )
                        st.rerun()

# Footer
st.markdown("---")
st.markdown("This tool helps you optimize your resume for ATS systems by analyzing keyword matches with job descriptions.")
