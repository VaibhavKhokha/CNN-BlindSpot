import streamlit as st
from PIL import Image
import torch
import sys
import os
import plotly.express as px
import numpy as np

sys.path.append(os.path.abspath("src"))
from model import load_model, get_probabilities
from image_processing import apply_grid_mask
from occlusion import calculate_sensitivity_matrix
from game import get_missions, check_mission_complete
from experiments import create_empty_log, log_action, convert_df_to_csv
from academy import render_academy_page

st.set_page_config(page_title="CNN Blind Spot", layout="wide")

@st.cache_resource
def init_model():
    return load_model()

model, preprocess, categories = init_model()

if 'masked_cells' not in st.session_state:
    st.session_state.masked_cells = set()
if 'mission_idx' not in st.session_state:
    st.session_state.mission_idx = 0
if 'exp_log' not in st.session_state:
    st.session_state.exp_log = create_empty_log()

def render_lab():
    st.title("CNN Blind Spot: Interactive Lab")
    
    missions = get_missions()
    current_mission = missions[st.session_state.mission_idx] if st.session_state.mission_idx < len(missions) else None

    img_path = "assets/images/dog.jpg"
    original_img = Image.open(img_path).convert("RGB")
    grid_size = 4

    orig_probs = get_probabilities(original_img, model, preprocess)
    target_class_id = torch.argmax(orig_probs).item()
    orig_score = orig_probs[target_class_id].item() * 100
    target_class_name = categories[target_class_id]

    with st.sidebar:
        st.header("Mission Log")
        move_count = len(st.session_state.masked_cells)
        if current_mission:
            st.subheader(f"Mission {current_mission.id}: {current_mission.title}")
            st.info(current_mission.description)
            st.markdown(f"** Learning Objective:** {current_mission.learning_objective}")
            st.markdown(f"** Related Theory:** {current_mission.chapter_link}")
            if current_mission.max_moves:
                st.write(f"**Moves Used:** {move_count} / {current_mission.max_moves}")
            else:
                st.write(f"**Moves Used:** {move_count}")
        else:
            st.success("All missions completed! Feel free to experiment in the Sandbox.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Interactive Image")
        display_img = original_img.copy()
        for (r, c) in st.session_state.masked_cells:
            display_img = apply_grid_mask(display_img, grid_size, r, c)
            
        st.image(display_img, use_container_width=True)
        
        for r in range(grid_size):
            cols = st.columns(grid_size)
            for c in range(grid_size):
                btn_lbl = f"R{r} C{c}"
                if (r, c) in st.session_state.masked_cells:
                    btn_lbl = f"[{btn_lbl}]"
                    
                if cols[c].button(btn_lbl, key=f"btn_{r}_{c}"):
                    if (r, c) in st.session_state.masked_cells:
                        st.session_state.masked_cells.remove((r, c))
                        action_str = "Unmask"
                    else:
                        st.session_state.masked_cells.add((r, c))
                        action_str = "Mask"
                    
                    temp_img = original_img.copy()
                    for (mr, mc) in st.session_state.masked_cells:
                        temp_img = apply_grid_mask(temp_img, grid_size, mr, mc)
                    new_probs = get_probabilities(temp_img, model, preprocess)
                    new_score = new_probs[target_class_id].item() * 100
                    m_title = current_mission.title if current_mission else "Sandbox"
                    
                    st.session_state.exp_log = log_action(
                        st.session_state.exp_log, m_title, target_class_name,
                        orig_score, action_str, f"R{r}C{c}", new_score
                    )
                    st.rerun()

        if st.button("Reset Image", type="primary"):
            st.session_state.masked_cells.clear()
            st.rerun()

    with col2:
        st.subheader("CNN Predictions")
        probs = get_probabilities(display_img, model, preprocess)
        current_score = probs[target_class_id].item() * 100
        
        top3_prob, top3_catid = torch.topk(probs, 3)
        for i in range(3):
            class_name = categories[top3_catid[i].item()]
            score = top3_prob[i].item() * 100
            st.progress(int(score), text=f"**{class_name}**: {score:.2f}%")
            
        if current_mission:
            is_complete, reason = check_mission_complete(current_mission, orig_score, current_score, move_count)
            if is_complete:
                st.sidebar.success(reason)
                if st.sidebar.button("Next Mission"):
                    st.session_state.mission_idx += 1
                    st.session_state.masked_cells.clear()
                    st.rerun()
            elif move_count > 0:
                if "exceeded" in reason.lower():
                    st.sidebar.error(reason)
                else:
                    st.sidebar.warning(reason)

        st.divider()
        st.subheader("AI X-Ray")
        
        st.info("**Prediction-Before-Reveal**: Before generating the heatmap, formulate a scientific hypothesis.")
        hypothesis = st.radio("Which area do you predict is most critical for the 'Samoyed' prediction?", 
                              ["The dog's face/head (Top-Left/Center)", "The dog's body/tail", "The grass/background"], index=None)
                              
        if st.button("Submit Hypothesis & Generate Heatmap", disabled=(hypothesis is None)):
            with st.spinner("Calculating sensitivity matrix... (running 16 inferences)"):
                matrix = calculate_sensitivity_matrix(original_img, model, preprocess, target_class_id, grid_size)
                
                st.success(f"Hypothesis recorded: **{hypothesis}**")
                st.markdown("Analyze the data below. The dark red regions (Positive D) caused the largest drop in confidence. Was your prediction right?")
                
                fig1 = px.imshow(matrix * 100, text_auto=".1f", color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
                                title="Signed Sensitivity D(i) (%)", labels=dict(color="Drop (%)"))
                st.plotly_chart(fig1, use_container_width=True)
                
                st.markdown("**Absolute Effect Heatmap:** Showing magnitude of change, ignoring whether it helped or hurt the score.")
                abs_matrix = np.abs(matrix)
                fig2 = px.imshow(abs_matrix * 100, text_auto=".1f", color_continuous_scale="Reds", 
                                title="Absolute Effect A(i) (%)", labels=dict(color="Magnitude (%)"))
                st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.header("Experiment Lab")
    st.dataframe(st.session_state.exp_log, use_container_width=True)
    if not st.session_state.exp_log.empty:
        csv_data = convert_df_to_csv(st.session_state.exp_log)
        st.download_button(label="Download CSV", data=csv_data, file_name="cnn_experiment_log.csv", mime="text/csv")

def render_assessment():
    st.title("Final Assessment")
    st.markdown("Test your knowledge of CNNs and Model Interpretability to complete your training.")
    
    questions = [
        {
            "q": "1. What is the primary purpose of a Convolutional Filter (Kernel)?",
            "opts": ["To delete random pixels to save memory.", "To slide over an image and mathematically detect spatial patterns like edges.", "To convert the image to grayscale."],
            "ans": 1,
            "exp": "Filters perform element-wise multiplication to detect specific spatial patterns."
        },
        {
            "q": "2. What does the Softmax function do at the end of the network?",
            "opts": ["It forces all output logits to sum to exactly 1.0 (100%), creating a probability distribution.", "It draws bounding boxes around objects.", "It removes all negative numbers from the feature map."],
            "ans": 0,
            "exp": "Softmax converts raw logits into readable probability scores."
        },
        {
            "q": "3. In our Occlusion Sensitivity formula (D = S_original - S_masked), what does a large POSITIVE number mean?",
            "opts": ["Masking the region increased the model's confidence.", "Masking the region caused a massive DROP in the model's confidence.", "The model is 100% accurate."],
            "ans": 1,
            "exp": "If D is positive, the Original score was much higher than the Masked score, meaning the region was critical to the prediction."
        },
        {
            "q": "4. If you mask the grass behind a dog and the confidence score goes UP, what happened?",
            "opts": ["The grass was distracting noise, so removing it helped the CNN focus on the dog's features.", "The CNN thinks grass is a dog.", "The formula broke."],
            "ans": 0,
            "exp": "Removing noisy background features can occasionally improve a model's confidence."
        },
        {
            "q": "5. True or False: If a heatmap strongly highlights a dog's face, it proves the CNN possesses human-like understanding of what a 'face' is.",
            "opts": ["True", "False"],
            "ans": 1,
            "exp": "False. It only proves those specific pixel arrangements mathematically triggered the highest activations. It does not prove cognitive understanding."
        }
    ]
    
    score = 0
    submitted = False
    
    with st.form("final_exam"):
        answers = []
        for i, item in enumerate(questions):
            st.markdown(f"**{item['q']}**")
            ans = st.radio("Select:", item['opts'], key=f"q_{i}", index=None)
            answers.append(ans)
            st.divider()
            
        submitted = st.form_submit_button("Submit Final Assessment")
        
    if submitted:
        for i, item in enumerate(questions):
            if answers[i] == item['opts'][item['ans']]:
                score += 1
                st.success(f"Question {i+1}: Correct! {item['exp']}")
            else:
                st.error(f"Question {i+1}: Incorrect. {item['exp']}")
                
        st.header(f"Your Score: {score} / {len(questions)}")
        if score == len(questions):
            st.balloons()
            st.success("Congratulations! You have mastered the CNN Blind Spot curriculum!")
        else:
            st.warning("Review the Learning Academy chapters and try again!")

# --- Main Application Router ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Learning Academy", "Interactive Lab", "Final Assessment"])

if page == "Learning Academy":
    render_academy_page(model, preprocess, categories)
elif page == "Interactive Lab":
    render_lab()
elif page == "Final Assessment":
    render_assessment()
