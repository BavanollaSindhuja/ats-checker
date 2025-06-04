import streamlit as st
import re
import PyPDF2
import docx2txt
import nltk
from nltk.corpus import stopwords
from openai import OpenAI
import os

# Initialize the app
def main():
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
            
            # Test connection
            try:
                models = client.models.list()
                available_models = [model.id for model in models.data]
                preferred_models = [
                    "openai/gpt-4o",
                    "openai/gpt-4-turbo",
                    "anthropic/claude-3-haiku",
                    "google/gemini-pro-1.5"
                ]
                
                selected_model = next((m for m in preferred_models if m in available_models), None)
                return True, client, selected_model if selected_model else "anthropic/claude-3-haiku"
            except Exception:
                return False, None, None
        return False, None, None
    except Exception:
        return False, None, None

def setup_streamlit_app(openrouter_available, openrouter_client, selected_model):
    st.set_page_config(page_title="ATS Score Checker", layout="wide")

    # Initialize session state
    session_defaults = {
        'resume_text': "",
        'job_description': "",
        'score': 0,
        'matching_keywords': [],
        'missing_keywords': [],
        'ai_recommendations': ""
    }
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

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
                openrouter_available = True
                
        st.warning("Free accounts have limited tokens. For full analysis, [upgrade your OpenRouter account](https://openrouter.ai/settings/credits)")

    # Main UI
    st.title("🎯 ATS Resume Score Checker")
    st.markdown("Upload your resume and paste a job description to check your ATS compatibility score.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📄 Upload Resume")
        uploaded_file = st.file_uploader("Choose file", type=["pdf", "docx", "txt"], key="file_uploader")
        if uploaded_file:
            st.session_state.resume_text = extract_text_from_file(uploaded_file)
        
        st.subheader("✏️ Or Paste Resume Text")
        resume_text = st.text_area("Paste resume text here", height=200, value=st.session_state.resume_text)

    with col2:
        st.subheader("📋 Job Description")
        job_desc = st.text_area("Paste job description here", height=300, value=st.session_state.job_description)
        if job_desc:
            st.session_state.job_description = job_desc

    if st.button("🔍 Check ATS Score", type="primary"):
        if not st.session_state.resume_text and not resume_text:
            st.error("Please upload or paste your resume")
        elif not st.session_state.job_description:
            st.error("Please paste the job description")
        else:
            with st.spinner("Analyzing..."):
                if resume_text:
                    st.session_state.resume_text = resume_text
                analyze_resume(ai_provider, openrouter_client, selected_model)

    display_results(ai_provider)

def extract_text_from_file(uploaded_file):
    try:
        if uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            return " ".join(page.extract_text() for page in pdf_reader.pages)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return docx2txt.process(uploaded_file)
        elif uploaded_file.type == "text/plain":
            return str(uploaded_file.read(), "utf-8")
        return ""
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return ""

def analyze_resume(ai_provider, openrouter_client, selected_model):
    score, matching, missing = calculate_ats_score(
        st.session_state.resume_text,
        st.session_state.job_description
    )
    
    st.session_state.score = score
    st.session_state.matching_keywords = matching
    st.session_state.missing_keywords = missing
    
    if ai_provider == "OpenRouter" and openrouter_client:
        # Check text length before sending to API
        if len(st.session_state.resume_text) < 5000 and len(st.session_state.job_description) < 5000:
            st.session_state.ai_recommendations = get_openrouter_recommendations(
                openrouter_client,
                st.session_state.resume_text,
                st.session_state.job_description,
                matching,
                missing,
                score,
                selected_model
            )
        else:
            st.session_state.ai_recommendations = "⚠️ Document too long for free analysis. Please reduce text size or upgrade your OpenRouter account."

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
        # Truncate long texts to stay within token limits
        truncated_resume = resume_text[:1000]
        truncated_job_desc = job_desc[:1000]
        
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": f"""
                Provide 3-4 specific recommendations to improve this resume for the job:
                
                Resume Excerpt: {truncated_resume}
                Job Description Excerpt: {truncated_job_desc}
                
                ATS Score: {score:.1f}%
                Matching Keywords: {', '.join(matching[:10])}
                Missing Keywords: {', '.join(missing[:10])}
                """
            }],
            max_tokens=500  # Strict token limit
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ API Error: {str(e)}"

def display_results(ai_provider):
    if st.session_state.score > 0:
        st.header("📊 Results")
        st.subheader(f"ATS Score: {st.session_state.score:.1f}%")
        
        # Score bar
        st.progress(int(st.session_state.score))
        
        # Results tabs
        tab1, tab2, tab3 = st.tabs(["🔍 Keywords", "💡 Tips", "🤖 AI Analysis"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("✅ Matching Keywords")
                st.write("\n".join(f"- {kw}" for kw in st.session_state.matching_keywords[:20]))
            with col2:
                st.subheader("❌ Missing Keywords")
                st.write("\n".join(f"- {kw}" for kw in st.session_state.missing_keywords[:20]))
        
        with tab2:
            st.subheader("Improvement Tips")
            if st.session_state.score < 50:
                st.write("1. Add more keywords from the Missing Keywords list")
                st.write("2. Highlight required skills in a dedicated skills section")
                st.write("3. Use exact phrases from the job description")
            elif st.session_state.score < 75:
                st.write("1. Quantify achievements with numbers/metrics")
                st.write("2. Reorder content to put most relevant experience first")
                st.write("3. Add industry-specific terminology")
            else:
                st.write("1. Check for typos and formatting consistency")
                st.write("2. Remove unrelated experience to shorten resume")
                st.write("3. Add portfolio links or certifications")
        
        with tab3:
            st.subheader("AI Recommendations")
            if ai_provider == "OpenRouter" and st.session_state.ai_recommendations:
                st.write(st.session_state.ai_recommendations)
            else:
                st.info("Enable OpenRouter in sidebar for AI suggestions")

if __name__ == "__main__":
    main()
