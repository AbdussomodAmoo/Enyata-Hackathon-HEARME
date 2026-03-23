import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
import pandas as pd

# --- INITIAL SETUP ---
st.set_page_config(page_title="BridgeLens | Universal Accessibility", layout="wide", page_icon="🤟")

# Initialize Session State
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'last_sign' not in st.session_state:
    st.session_state['last_sign'] = "None"
if 'transcription' not in st.session_state:
    st.session_state['transcription'] = []

# --- UNIVERSAL LOGIC: THE SIGN GLOSS CONVERTER ---
def convert_to_nsl_gloss(text):
    # Simplifies English to NSL Keywords (Glosses) for immediate visual understanding
    words = text.upper().split()
    stop_words = ["THE", "IS", "AM", "A", "AN", "ARE", "OF", "TO", "FOR"]
    glosses = [w for w in words if w not in stop_words]
    return glosses

# --- UI BRANDING ---
st.title("🤟 BridgeLens Universal Accessibility")
st.markdown("### *AI-Powered Gateway for Social, Health, and Financial Inclusion*")
st.divider()

# --- SIDEBAR: MISSION & LISTENER ---
with st.sidebar:
    st.header("The Mission")
    st.warning("**Problem:** 1M+ Nigerians are 'Digitally Muted' in hospitals and banks.")
    st.success("**Solution:** AI-driven voice and visual interpretation using NSL.")
    st.write("---")
    
    st.header("🔊 Universal Listener")
    st.write("Converts nearby speech to Sign Glosses.")
    if st.button("🔴 Listen for Speech"):
        mock_speech = "The doctor says you need to pay 5000 Naira for the medicine"
        st.session_state['transcription'] = convert_to_nsl_gloss(mock_speech)
        st.success("Speech Captured!")

    if st.session_state['transcription']:
        st.write("**NSL Gloss Translation:**")
        # Visual icons represent the conceptual 'signs' rather than English words
        for word in st.session_state['transcription']:
            st.button(f"🆔 {word}", key=f"sidebar_{word}")

# --- THE TABBED INTERFACE ---
tab_daily, tab_health, tab_pay, tab_media = st.tabs([
    "🌍 Daily Interaction", "🏥 Medical Visit", "💳 Financial Inclusion", "📺 Media Access"
])

# --- 1. DAILY INTERACTION ---
with tab_daily:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Live Sign Translation")
        st.camera_input("Sign to the camera...", key="daily_cam")
        if st.button("Translate Sign"):
            st.session_state['last_sign'] = "HELP"
            st.session_state['history'].append({"Time": time.strftime("%H:%M:%S"), "Action": "Help Requested", "Sector": "Social"})
            st.toast("Alert Sent!", icon="🚨")
    with col2:
        st.subheader("Status")
        st.write(f"**Intent:** {st.session_state['last_sign']}")
        st.progress(85, text="AI Confidence")
        st.dataframe(pd.DataFrame(st.session_state['history']).tail(5), use_container_width=True)

# --- 2. MEDICAL VISIT ---
with tab_health:
    st.header("Clinical Interaction Module")
    h_col1, h_col2 = st.columns(2)
    with h_col1:
        st.subheader("Symptom Identification")
        st.image("https://via.placeholder.com/400x300.png?text=MediaPipe+Overlay", caption="Sign: 'PAIN'")
        # WHY ACTIVE DIAGNOSTICS: It allows the Deaf user to 'confirm' what the AI thinks they signed.
        # It prevents medical errors by providing a visual 'receipt' of their symptoms.
        st.multiselect("Confirm Detected Symptoms", ["Fever", "Pain", "Cough", "Malaria"], default=["Pain"])
    with h_col2:
        st.subheader("Verification & Notes")
        if st.button("Verify Medication (Blockchain)"):
            with st.status("Scanning Drug Ledger..."):
                time.sleep(1.5)
                st.success("Authentic: Batch #0x8823 Verified")
        st.text_area("Doctor's Report", "Patient indicates acute pain in the upper region...", height=150)

# --- 3. FINANCIAL INCLUSION ---
with tab_pay:
    st.header("Secure Banking Gateway")
    st.info("Authorized by Interswitch Identity Rails")
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        st.write("**Biometric Sign Auth**")
        st.warning("Sign your 'Private Gesture' to authorize.")
        amt = st.number_input("Amount (NGN)", value=5000)
        acc = st.text_input("Recipient Account", "0123456789")
        if st.button("Authorize via Interswitch"):
            with st.status("Verifying via Interswitch...") as s:
                time.sleep(1.5)
                st.write("Checking `AccountNameInquiry`...")
                time.sleep(1)
                s.update(label="Transfer Successful!", state="complete")
            st.balloons()
    with p_col2:
        st.metric("Daily Limit", "N100,000", "-N5,000")
        st.write("**Why this is Secure:**")
        st.write("Your sign is your password. It uses behavioral biometrics that cannot be 'phished' like a PIN.")

# --- 4. MEDIA ACCESS ---
with tab_media:
    st.header("Digital Content Bridge")
    m_col1, m_col2 = st.columns([3, 2])
    with m_col1:
        st.video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    with m_col2:
        if st.button("✨ Generate Sign Gloss Summary"):
            with st.status("Processing Audio..."):
                time.sleep(2)
                st.button("💰 MONEY", type="primary")
                st.button("🏥 HEALTH", type="primary")
                st.success("Summary Generated")

st.divider()
st.caption("BridgeLens | Enyata x Interswitch Buildathon 2026")
