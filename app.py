import streamlit as st
import re
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import PyPDF2
import docx2txt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from openai import OpenAI
import os
from dotenv import load_dotenv

# Initialize the app
def main():
    # Load environment variables from .env file
    load_dotenv()

    # Download necessary NLTK data
    nltk.download('punkt')
    nltk.download('stopwords')

    # Configure OpenRouter API
    openrouter_available, openrouter_client, selected_model = configure_openrouter()

    # Set up the Streamlit app
    setup_streamlit_app(openrouter_available, openrouter_client, selected_model)

def configure_openrouter():
    try:
        OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
        if OPENROUTER_API_KEY:
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
            )
            
            # Test the connection with a simple models request
            try:
                models = client.models.list()
                available_models = [model.id for model in models.data]
                preferred_models = [
                    "openai/gpt-4o",
                    "openai/gpt-4-turbo",
                    "anthropic/claude-3-opus",
                    "google/gemini-pro-1.5"
                ]
                
                # Find the first available preferred model
                selected_model = next((m for m in preferred_models if m in available_models), None)
                
                if selected_model:
                    return True, client, selected_model
                return False, None, None
            except Exception as test_error:
                print(f"OpenRouter connection test failed: {test_error}")
                return False, None, None
        return False, None, None
    except Exception as e:
        print(f"Error configuring OpenRouter: {e}")
        return False, None, None

def setup_streamlit_app(openrouter_available, openrouter_client, selected_model):
    st.set_page_config(page_title="ATS Score Checker", layout="wide")

    # Initialize session state
    if 'resume_text' not in st.session_state:
        st.session_state.resume_text = ""
    if 'job_description' not in st.session_state:
        st.session_state.job_description = ""
    if 'score' not in st.session_state:
        st.session_state.score = 0
    if 'matching_keywords' not in st.session_state:
        st.session_state.matching_keywords = []
    if 'missing_keywords' not in st.session_state:
        st.session_state.missing_keywords = []
    if 'ai_recommendations' not in st.session_state:
        st.session_state.ai_recommendations = ""

    # Sidebar configuration
    with st.sidebar:
        st.header("AI Settings")
        ai_provider = st.selectbox(
            "Choose AI Provider:",
            ["None", "OpenRouter"],
            help="Select OpenRouter for AI-powered recommendations"
        )
        
        if ai_provider == "OpenRouter":
            openrouter_api_key = st.text_input("Enter OpenRouter API Key", type="password")
            if openrouter_api_key:
                os.environ["OPENROUTER_API_KEY"] = openrouter_api_key
                openrouter_client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=openrouter_api_key,
                )
                st.success("OpenRouter API key configured!")
                # Update availability after key entry
                openrouter_available = True

    # Main application UI
    st.title("🎯 ATS Resume Score Checker")
    st.markdown("Upload your resume and paste a job description to check your ATS compatibility score.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📄 Upload Resume")
        uploaded_file = st.file_uploader("Choose file", type=["pdf", "docx", "txt"])
        if uploaded_file:
            st.session_state.resume_text = extract_text_from_file(uploaded_file)
            st.success("Resume uploaded!")
        
        st.subheader("✏️ Or Paste Resume Text")
        resume_text = st.text_area("Paste resume text here", height=200)
        if resume_text:
            st.session_state.resume_text = resume_text

    with col2:
        st.subheader("📋 Job Description")
        job_desc = st.text_area("Paste job description here", height=300)
        if job_desc:
            st.session_state.job_description = job_desc

    if st.button("🔍 Check ATS Score", type="primary"):
        if not st.session_state.resume_text:
            st.error("Please upload or paste your resume")
        elif not st.session_state.job_description:
            st.error("Please paste the job description")
        else:
            with st.spinner("Analyzing..."):
                analyze_resume(ai_provider, openrouter_client, selected_model)

    display_results(ai_provider)

def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        return " ".join(page.extract_text() for page in pdf_reader.pages)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return docx2txt.process(uploaded_file)
    elif uploaded_file.type == "text/plain":
        return str(uploaded_file.read(), "utf-8")
    return None

def analyze_resume(ai_provider, openrouter_client, selected_model):
    score, matching, missing = calculate_ats_score(
        st.session_state.resume_text,
        st.session_state.job_description
    )
    
    st.session_state.score = score
    st.session_state.matching_keywords = matching
    st.session_state.missing_keywords = missing
    
    if ai_provider == "OpenRouter" and openrouter_client:
        st.session_state.ai_recommendations = get_openrouter_recommendations(
            openrouter_client,
            st.session_state.resume_text,
            st.session_state.job_description,
            matching,
            missing,
            score,
            selected_model
        )

def calculate_ats_score(resume_text, job_description):
    if not resume_text or not job_description:
        return 0, [], []
    
    job_keywords = extract_keywords(job_description)
    resume_tokens = set(preprocess_text(resume_text))
    
    matching = [kw for kw in job_keywords if kw in resume_tokens]
    missing = [kw for kw in job_keywords if kw not in resume_tokens]
    
    score = len(matching) / len(job_keywords) * 100 if job_keywords else 0
    return score, matching, missing

def extract_keywords(text):
    tokens = preprocess_text(text)
    freq_dist = nltk.FreqDist(tokens)
    return [word for word, freq in freq_dist.most_common(30) if len(word) > 3]

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

def get_openrouter_recommendations(client, resume_text, job_desc, matching, missing, score, model):
    try:
        # Reduce input length significantly
        truncated_resume = resume_text[:1000]  # Only first 1000 characters
        truncated_job_desc = job_desc[:1000]   # Only first 1000 characters
        
        completion = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": f"""
                Analyze this resume against the job description:
                
                Resume: {truncated_resume}
                Job Description: {truncated_job_desc}
                
                ATS Score: {score:.1f}%
                Matching Keywords: {', '.join(matching)}
                Missing Keywords: {', '.join(missing)}
                
                Provide 3-4 specific recommendations to improve the resume.
                """
            }],
            max_tokens=1000  # Limit output tokens
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"
        
def display_results(ai_provider):
    if st.session_state.score > 0:
        st.header("📊 Results")
        
        # Score visualization
        st.subheader(f"ATS Score: {st.session_state.score:.1f}%")
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.barh([0], [100], color='lightgray')
        ax.barh([0], [st.session_state.score], color=get_score_color(st.session_state.score))
        ax.text(50, 0, f"{st.session_state.score:.1f}%", ha='center', va='center', fontsize=14, weight='bold')
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_yticks([])
        st.pyplot(fig)
        
        # Results tabs
        tab1, tab2, tab3 = st.tabs(["🔍 Keywords", "💡 Tips", "🤖 AI Analysis"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("✅ Matching Keywords")
                st.write("\n".join(f"- {kw}" for kw in st.session_state.matching_keywords))
            with col2:
                st.subheader("❌ Missing Keywords")
                st.write("\n".join(f"- {kw}" for kw in st.session_state.missing_keywords))
        
        with tab2:
            st.subheader("Improvement Tips")
            if st.session_state.score < 50:
                st.write("Your resume needs significant improvement:")
                st.write("- Add more relevant keywords from the job description")
                st.write("- Highlight your skills that match the requirements")
            elif st.session_state.score < 75:
                st.write("Good start, but could be better:")
                st.write("- Reorganize content to emphasize key qualifications")
                st.write("- Quantify your achievements where possible")
            else:
                st.write("Excellent match! Consider minor tweaks:")
                st.write("- Ensure consistent formatting throughout")
                st.write("- Double-check for any typos or errors")
        
        with tab3:
            st.subheader("AI Recommendations")
            if ai_provider == "OpenRouter" and st.session_state.ai_recommendations:
                st.write(st.session_state.ai_recommendations)
            else:
                st.info("Select OpenRouter in the sidebar for detailed AI analysis")

def get_score_color(score):
    if score < 25: return 'red'
    elif score < 50: return 'orange'
    elif score < 75: return 'yellowgreen'
    return 'green'

if __name__ == "__main__":
    main()