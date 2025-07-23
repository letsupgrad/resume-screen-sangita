import streamlit as st
from streamlit_chat import message
import PyPDF2  # For PDF processing
import docx2txt  # For DOCX processing
import tempfile
import os
from datetime import datetime
import json
import pandas as pd
from typing import Dict, List, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re

# Initialize session state
def init_session_state():
    if 'candidates' not in st.session_state:
        st.session_state.candidates = {}
    
    if 'active_candidate' not in st.session_state:
        st.session_state.active_candidate = None
    
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your AI recruitment assistant. Let's start with some basic questions. What's your name?"}
        ]
    
    if 'hr_mode' not in st.session_state:
        st.session_state.hr_mode = False
    
    if 'job_description' not in st.session_state:
        st.session_state.job_description = ""
    
    if 'comparison_mode' not in st.session_state:
        st.session_state.comparison_mode = False

def create_new_candidate(name: str) -> dict:
    """Create a new candidate profile"""
    return {
        'name': name,
        'position': None,
        'responses': [],
        'current_question': 'name',
        'status': 'start',
        'scores': {
            'overall': 0,
            'experience': 0,
            'skills': 0,
            'culture_fit': 0,
            'education': 0,
            'keywords_match': 0
        },
        'resume_text': None,
        'resume_uploaded': False,
        'resume_analysis': None,
        'file_name': None,
        'upload_timestamp': datetime.now().isoformat()
    }

# Enhanced resume analysis function
def extract_info_from_resume(resume_text: str) -> dict:
    """Extract comprehensive information from resume text"""
    info = {
        'skills': [],
        'experience_years': 0,
        'education': [],
        'certifications': [],
        'projects': [],
        'companies': [],
        'keywords': []
    }
    
    # Technical skills keywords
   
    technical_skills = [
        'python', 'java', 'javascript', 'sql', 'html', 'css', 'react', 'angular', 'vue',
        'node.js', 'django', 'flask', 'spring', 'mongodb', 'postgresql', 'mysql','nosql'
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'git', 'jenkins', 'terraform',
        'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'scikit-learn',
        'pandas', 'numpy', 'matplotlib', 'tableau', 'power bi', 'excel', 'r',
        'c++', 'c#', '.net', 'php', 'ruby', 'go', 'rust', 'swift', 'kotlin', 'AI', 'ML'
        'agile', 'scrum', 'devops', 'ci/cd', 'microservices', 'rest api', 'graphql'
        'Agile methodologies' , 'TensorFlow', 'Keras', 'hadoop', 'spark',  'Google Vertex AI'
    ]
    
    
    # Soft skills keywords
    soft_skills = [
        'leadership', 'communication', 'teamwork', 'problem solving', 'analytical',
        'project management', 'time management', 'adaptability', 'creativity',
        'critical thinking', 'collaboration', 'mentoring', 'training'
    ]
    
    # Education keywords
    education_keywords = [
        'bachelor', 'master', 'phd', 'doctorate', 'degree', 'university', 'college',
        'computer science', 'engineering', 'mathematics', 'statistics', 'mba'
    ]
    
    # Certification keywords
    cert_keywords = [
        'aws certified', 'azure certified', 'google cloud', 'cisco', 'microsoft certified',
        'pmp', 'scrum master', 'oracle certified', 'comptia', 'cissp'
    ]
    
    resume_lower = resume_text.lower()
    
    # Extract skills
    for skill in technical_skills + soft_skills:
        if skill in resume_lower:
            info['skills'].append(skill.title())
    
    # Extract education
    for edu in education_keywords:
        if edu in resume_lower:
            info['education'].append(edu.title())
    
    # Extract certifications
    for cert in cert_keywords:
        if cert in resume_lower:
            info['certifications'].append(cert.title())
    
    # Extract experience years (simple pattern matching)
    experience_patterns = [
        r'(\d+)\+?\s*years?\s*of\s*experience',
        r'(\d+)\+?\s*years?\s*experience',
        r'experience\s*:\s*(\d+)\+?\s*years?'
    ]
    
    for pattern in experience_patterns:
        matches = re.findall(pattern, resume_lower)
        if matches:
            info['experience_years'] = max([int(match) for match in matches])
            break
    
    # Extract company names (simple approach - look for common company indicators)
    company_indicators = ['inc', 'ltd', 'corp', 'llc', 'technologies', 'solutions', 'systems']
    lines = resume_text.split('\n')
    for line in lines:
        line_lower = line.lower()
        if any(indicator in line_lower for indicator in company_indicators):
            if len(line.strip()) < 50:  # Assume company names are not too long
                info['companies'].append(line.strip())
    
    return info

def calculate_resume_score(resume_text: str, job_description: str) -> dict:
    """Calculate comprehensive resume score based on job description"""
    
    if not job_description.strip():
        return {
            'overall_score': 0,
            'skills_match': 0,
            'experience_score': 0,
            'education_score': 0,
            'keyword_match': 0,
            'detailed_analysis': {
                'matched_skills': [],
                'missing_skills': [],
                'experience_level': 'Unknown',
                'education_match': False
            }
        }
    
    resume_info = extract_info_from_resume(resume_text)
    job_desc_lower = job_description.lower()
    resume_lower = resume_text.lower()
    
    # Extract required skills from job description
    job_skills = []
    skill_keywords = [
        'python', 'java', 'javascript', 'sql', 'html', 'css', 'react', 'angular',
        'machine learning', 'data science', 'aws', 'azure', 'docker', 'kubernetes',
        'project management', 'leadership', 'communication', 'teamwork'
    ]
    
    for skill in skill_keywords:
        if skill in job_desc_lower:
            job_skills.append(skill)
    
    # Calculate skills match
    matched_skills = []
    for skill in job_skills:
        if skill in resume_lower:
            matched_skills.append(skill)
    
    skills_match_score = len(matched_skills) / len(job_skills) if job_skills else 0
    
    # Calculate experience score
    required_exp_pattern = r'(\d+)\+?\s*years?\s*of\s*experience'
    required_exp_matches = re.findall(required_exp_pattern, job_desc_lower)
    required_exp = int(required_exp_matches[0]) if required_exp_matches else 0
    
    experience_score = min(1.0, resume_info['experience_years'] / max(required_exp, 1)) if required_exp > 0 else 0.5
    
    # Calculate education score
    education_score = 0.5
    if 'bachelor' in job_desc_lower or 'degree' in job_desc_lower:
        if any('bachelor' in edu.lower() or 'degree' in edu.lower() for edu in resume_info['education']):
            education_score = 0.8
        if any('master' in edu.lower() for edu in resume_info['education']):
            education_score = 1.0
    
    # Calculate keyword match
    job_keywords = set(job_desc_lower.split())
    resume_keywords = set(resume_lower.split())
    keyword_match = len(job_keywords.intersection(resume_keywords)) / len(job_keywords) if job_keywords else 0
    
    # Calculate overall score
    overall_score = (
        skills_match_score * 0.4 +
        experience_score * 0.3 +
        education_score * 0.2 +
        keyword_match * 0.1
    )
    
    return {
        'overall_score': overall_score,
        'skills_match': skills_match_score,
        'experience_score': experience_score,
        'education_score': education_score,
        'keyword_match': keyword_match,
        'detailed_analysis': {
            'matched_skills': matched_skills,
            'missing_skills': [skill for skill in job_skills if skill not in matched_skills],
            'experience_level': f"{resume_info['experience_years']} years" if resume_info['experience_years'] > 0 else "Not specified",
            'education_match': education_score > 0.5,
            'total_skills_found': len(resume_info['skills']),
            'certifications': resume_info['certifications']
        }
    }

def create_comparison_charts(candidates: dict) -> tuple:
    """Create comparison charts for multiple candidates"""
    
    if not candidates:
        return None, None
    
    # Prepare data for comparison
    candidate_names = []
    overall_scores = []
    skills_scores = []
    experience_scores = []
    education_scores = []
    keyword_scores = []
    
    for candidate_id, candidate in candidates.items():
        if candidate.get('resume_analysis'):
            candidate_names.append(candidate['name'])
            analysis = candidate['resume_analysis']
            overall_scores.append(analysis['overall_score'] * 100)
            skills_scores.append(analysis['skills_match'] * 100)
            experience_scores.append(analysis['experience_score'] * 100)
            education_scores.append(analysis['education_score'] * 100)
            keyword_scores.append(analysis['keyword_match'] * 100)
    
    if not candidate_names:
        return None, None
    
    # Overall comparison bar chart
    fig_overall = go.Figure(data=[
        go.Bar(
            x=candidate_names,
            y=overall_scores,
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'][:len(candidate_names)],
            text=[f"{score:.1f}%" for score in overall_scores],
            textposition='auto',
        )
    ])
    fig_overall.update_layout(
        title="Overall Candidate Comparison",
        yaxis_title="Score (%)",
        xaxis_title="Candidates",
        height=400
    )
    
    # Detailed comparison radar chart
    fig_radar = go.Figure()
    
    categories = ['Skills Match', 'Experience', 'Education', 'Keywords']
    
    for i, name in enumerate(candidate_names):
        fig_radar.add_trace(go.Scatterpolar(
            r=[skills_scores[i], experience_scores[i], education_scores[i], keyword_scores[i]],
            theta=categories,
            fill='toself',
            name=name,
            line_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'][i % 5]
        ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Detailed Skills Comparison",
        height=500
    )
    
    return fig_overall, fig_radar

def create_candidate_table(candidates: dict) -> pd.DataFrame:
    """Create a comparison table for all candidates"""
    
    data = []
    for candidate_id, candidate in candidates.items():
        if candidate.get('resume_analysis'):
            analysis = candidate['resume_analysis']
            detailed = analysis['detailed_analysis']
            
            data.append({
                'Name': candidate['name'],
                'Overall Score': round(analysis['overall_score']*100, 1),
                'Skills Match': round(analysis['skills_match']*100, 1),
                'Experience': detailed['experience_level'],
                'Education Match': "Yes" if detailed['education_match'] else "No",
                'Total Skills': detailed['total_skills_found'],
                'Certifications': len(detailed['certifications']),
                'Upload Date': candidate['upload_timestamp'][:10]
            })
    
    return pd.DataFrame(data)

def process_multiple_resumes(uploaded_files: List, job_description: str) -> dict:
    """Process multiple resume files and return candidate data"""
    
    processed_candidates = {}
    
    for uploaded_file in uploaded_files:
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Extract text based on file type
            if uploaded_file.type == "application/pdf":
                with open(tmp_file_path, "rb") as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    resume_text = "\n".join([page.extract_text() for page in pdf_reader.pages])
            else:
                resume_text = docx2txt.process(tmp_file_path)
            
            # Extract candidate name (first line or filename)
            first_line = resume_text.split('\n')[0].strip()
            candidate_name = first_line if len(first_line) < 50 else uploaded_file.name.split('.')[0]
            
            # Create candidate ID
            candidate_id = f"candidate_{len(processed_candidates) + 1}"
            
            # Create candidate profile
            candidate = create_new_candidate(candidate_name)
            candidate['resume_text'] = resume_text
            candidate['resume_uploaded'] = True
            candidate['file_name'] = uploaded_file.name
            
            # Calculate resume score
            if job_description:
                resume_scores = calculate_resume_score(resume_text, job_description)
                candidate['resume_analysis'] = resume_scores
                
                # Update candidate scores
                candidate['scores'].update({
                    'overall': resume_scores['overall_score'],
                    'skills': resume_scores['skills_match'],
                    'experience': resume_scores['experience_score'],
                    'education': resume_scores['education_score'],
                    'keywords_match': resume_scores['keyword_match']
                })
            
            processed_candidates[candidate_id] = candidate
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        finally:
            if 'tmp_file_path' in locals():
                os.unlink(tmp_file_path)
    
    return processed_candidates

def analyze_response(text: str, question_type: str, has_resume: bool = False) -> dict:
    """Analyze candidate response with resume context"""
    analysis = {
        "sentiment": "positive",
        "keywords": [],
        "quality_score": 0.85,
        "question_type": question_type
    }
    
    if has_resume and st.session_state.active_candidate:
        candidate = st.session_state.candidates[st.session_state.active_candidate]
        if candidate.get('resume_text'):
            resume_info = extract_info_from_resume(candidate['resume_text'])
            analysis['resume_context'] = resume_info
            analysis['quality_score'] = min(1.0, analysis['quality_score'] + 0.1)
    
    return analysis

def generate_next_question(candidate_id: str) -> str:
    """Generate next question based on current state and resume availability"""
    
    candidate = st.session_state.candidates[candidate_id]
    current_question = candidate['current_question']
    
    if current_question == 'name':
        if candidate.get('resume_uploaded'):
            return "What position is this candidate applying for?"
        return "What position are you applying for?"
    
    elif current_question == 'position':
        candidate['position'] = candidate['responses'][-1]['answer']
        
        if candidate.get('resume_uploaded'):
            resume_info = extract_info_from_resume(candidate['resume_text'])
            return (f"I see experience with {', '.join(resume_info['skills'][:3])} in the resume. "
                   "Can you tell me about your most recent relevant project?")
        return "Tell me about your relevant experience for this role."
    
    elif current_question == 'experience':
        if candidate.get('resume_uploaded'):
            return "How would this role align with your career goals?"
        return "What are your salary expectations?"
    
    return "Thank you for your responses. We'll review your application and get back to you."

# Initialize the app
init_session_state()

# HR Mode Toggle
st.sidebar.title("HR Controls")
st.session_state.hr_mode = st.sidebar.checkbox("HR Mode", value=st.session_state.hr_mode)

if st.session_state.hr_mode:
    st.sidebar.subheader("Job Description")
    st.session_state.job_description = st.sidebar.text_area(
        "Enter Job Description",
        value=st.session_state.job_description,
        height=150,
        help="Enter the job description to compare against resumes"
    )
    
    st.sidebar.subheader("Multiple Resume Upload")
    
    # Multiple file uploader
    uploaded_files = st.sidebar.file_uploader(
        "Upload Candidate Resumes", 
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key="hr_multi_resume_upload"
    )
    
    if uploaded_files:
        if st.sidebar.button("Process All Resumes"):
            with st.spinner("Processing resumes..."):
                processed_candidates = process_multiple_resumes(uploaded_files, st.session_state.job_description)
                
                # Update session state
                st.session_state.candidates.update(processed_candidates)
                
                st.sidebar.success(f"Successfully processed {len(processed_candidates)} resumes!")
                
                # Enable comparison mode
                st.session_state.comparison_mode = True
    
    # Candidate selection for individual chat
    if st.session_state.candidates:
        st.sidebar.subheader("Select Candidate for Chat")
        candidate_options = {cid: candidate['name'] for cid, candidate in st.session_state.candidates.items()}
        
        selected_candidate = st.sidebar.selectbox(
            "Choose candidate",
            options=list(candidate_options.keys()),
            format_func=lambda x: candidate_options[x],
            key="candidate_selector"
        )
        
        if selected_candidate != st.session_state.active_candidate:
            st.session_state.active_candidate = selected_candidate
            candidate = st.session_state.candidates[selected_candidate]
            
            # Update chat messages for selected candidate
            st.session_state.messages = [
                {"role": "assistant", "content": f"Hello {candidate['name']}! Let's continue with your interview."},
                {"role": "assistant", "content": "What position are you applying for?"}
            ]
            
            candidate['current_question'] = 'position'
    
    # Comparison mode toggle
    if st.session_state.candidates:
        st.session_state.comparison_mode = st.sidebar.checkbox(
            "Comparison Mode", 
            value=st.session_state.comparison_mode,
            help="Enable to compare all candidates"
        )

# Main interface
st.title("AI Recruitment Assistant")

# Comparison Dashboard
if st.session_state.hr_mode and st.session_state.comparison_mode and st.session_state.candidates:
    st.header("📊 Candidate Comparison Dashboard")
    
    # Summary metrics
    total_candidates = len(st.session_state.candidates)
    analyzed_candidates = len([c for c in st.session_state.candidates.values() if c.get('resume_analysis')])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Candidates", total_candidates)
    with col2:
        st.metric("Analyzed Candidates", analyzed_candidates)
    with col3:
        if analyzed_candidates > 0:
            avg_score = sum([c['resume_analysis']['overall_score'] for c in st.session_state.candidates.values() if c.get('resume_analysis')]) / analyzed_candidates
            st.metric("Average Score", f"{avg_score*100:.1f}%")
    
    # Comparison charts
    if analyzed_candidates > 0:
        fig_overall, fig_radar = create_comparison_charts(st.session_state.candidates)
        
        if fig_overall and fig_radar:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_overall, use_container_width=True)
            with col2:
                st.plotly_chart(fig_radar, use_container_width=True)
    
    # Candidate comparison table
    st.subheader("📋 Detailed Candidate Comparison")
    
    if analyzed_candidates > 0:
        df = create_candidate_table(st.session_state.candidates)
        
        # Create a styled dataframe with proper formatting
        def format_percentage(val):
            if isinstance(val, (int, float)) and val <= 100:
                return f"{val}%"
            return val
        
        # Apply formatting and styling
        styled_df = df.style.format({
            'Overall Score': format_percentage,
            'Skills Match': format_percentage
        }).background_gradient(
            subset=['Overall Score', 'Skills Match'], 
            cmap='RdYlGn',
            vmin=0, 
            vmax=100
        ).set_properties(**{
            'text-align': 'center'
        })
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Download comparison report
        if st.button("📥 Download Comparison Report (JSON)"):
            comparison_report = {
                "comparison_timestamp": datetime.now().isoformat(),
                "job_description": st.session_state.job_description,
                "total_candidates": total_candidates,
                "candidates": st.session_state.candidates,
                "summary": {
                    "average_score": avg_score if analyzed_candidates > 0 else 0,
                    "top_candidate": max(st.session_state.candidates.items(), 
                                       key=lambda x: x[1].get('resume_analysis', {}).get('overall_score', 0))[1]['name'] if analyzed_candidates > 0 else None
                }
            }
            
            st.download_button(
                label="Download Report",
                data=json.dumps(comparison_report, indent=2),
                file_name=f"candidate_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        # Download candidate data as CSV
        if st.button("📥 Download Candidate Data (CSV)"):
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"candidate_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

    # Individual candidate details
    st.subheader("👤 Individual Candidate Details")
    
    if st.session_state.candidates:
        candidate_tabs = st.tabs([candidate['name'] for candidate in st.session_state.candidates.values()])
        
        for i, (candidate_id, candidate) in enumerate(st.session_state.candidates.items()):
            with candidate_tabs[i]:
                if candidate.get('resume_analysis'):
                    analysis = candidate['resume_analysis']
                    detailed = analysis['detailed_analysis']
                    
                    # Individual scores
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Overall Score", f"{analysis['overall_score']*100:.1f}%")
                    with col2:
                        st.metric("Skills Match", f"{analysis['skills_match']*100:.1f}%")
                    with col3:
                        st.metric("Experience", detailed['experience_level'])
                    with col4:
                        st.metric("Total Skills", detailed['total_skills_found'])
                    
                    # Skills breakdown
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Matched Skills:**")
                        for skill in detailed['matched_skills']:
                            st.success(f"✓ {skill.title()}")
                    
                    with col2:
                        st.write("**Missing Skills:**")
                        for skill in detailed['missing_skills']:
                            st.error(f"✗ {skill.title()}")
                    
                    # Certifications
                    if detailed['certifications']:
                        st.write("**Certifications:**")
                        for cert in detailed['certifications']:
                            st.info(f"🏆 {cert}")
                else:
                    st.info("Resume analysis not available for this candidate.")

# Chat interface (when not in comparison mode or when candidate is selected)
if not st.session_state.comparison_mode or not st.session_state.hr_mode:
    # Display chat messages
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "assistant":
            message(msg["content"], is_user=False, key=f"assistant_{i}")
        else:
            message(msg["content"], is_user=True, key=f"user_{i}")

    # Process user input
    def process_user_input():
        user_input = st.session_state.user_input
        
        if user_input.strip():
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Handle candidate selection
            if st.session_state.hr_mode and st.session_state.active_candidate:
                candidate_id = st.session_state.active_candidate
                candidate = st.session_state.candidates[candidate_id]
            else:
                # Create new candidate for non-HR mode
                if not st.session_state.candidates:
                    candidate_id = "candidate_1"
                    st.session_state.candidates[candidate_id] = create_new_candidate("Unknown")
                    st.session_state.active_candidate = candidate_id
                candidate = st.session_state.candidates[st.session_state.active_candidate]
            
            # Get current state
            current_question = candidate['current_question']
            
            # Analyze response
            analysis = analyze_response(
                user_input, 
                current_question,
                candidate.get('resume_uploaded', False)
            )
            
            # Store response and analysis
            candidate['responses'].append({
                "question": current_question,
                "answer": user_input,
                "analysis": analysis,
                "timestamp": datetime.now().isoformat()
            })
            
            # Update specific fields
            if current_question == 'name':
                candidate['name'] = user_input
            elif current_question == 'position':
                candidate['position'] = user_input
            
            # Generate next question
            next_question = generate_next_question(st.session_state.active_candidate)
            
            # Update current question
            if "thank you" in next_question.lower():
                candidate['status'] = 'complete'
            else:
                # Simple logic to determine the next question key. This might need refinement.
                # For now, we'll just set it to a simplified version of the question.
                candidate['current_question'] = next_question.split('?')[0].split('.')[0].strip().lower().replace(' ', '_')
            
            # Add assistant response to chat
            st.session_state.messages.append({"role": "assistant", "content": next_question})
            
            # Clear input box
            st.session_state.user_input = ""

    # Input box for candidate responses
    if (not st.session_state.hr_mode or 
        (st.session_state.active_candidate and 
         st.session_state.candidates[st.session_state.active_candidate].get('current_question'))):
        st.text_input("Your response:", key="user_input", on_change=process_user_input)

# Sidebar information
if st.session_state.candidates:
    st.sidebar.subheader("📊 Current Session Summary")
    st.sidebar.write(f"**Total Candidates:** {len(st.session_state.candidates)}")
    
    if st.session_state.active_candidate:
        active_candidate = st.session_state.candidates[st.session_state.active_candidate]
        st.sidebar.write(f"**Active Candidate:** {active_candidate['name']}")
        
        if active_candidate.get('resume_analysis'):
            overall_score = active_candidate['resume_analysis']['overall_score']
            st.sidebar.progress(overall_score)
            st.sidebar.write(f"Score: {overall_score*100:.1f}%")
