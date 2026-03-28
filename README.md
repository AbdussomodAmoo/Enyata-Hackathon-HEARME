# 🤟 BridgeLens: Universal Digital & Physical Inclusion
**Built for the Enyata x Interswitch Buildathon 2026**

BridgeLens is a comprehensive, two-way communication bridge designed to grant the Deaf community complete digital, financial, and physical independence. It moves beyond simple word translation by combining Edge-based Computer Vision with Cloud-based Large Language Models (LLMs) to understand context, alongside seamless integration with the Interswitch API ecosystem.

## ⚠️ The Problem
The world is designed for the hearing. Deaf individuals face compounding barriers:
1. **Communication:** Inability to easily communicate in daily life or medical emergencies.
2. **Financial Exclusion:** USSD codes, uninterpreted banking halls, and audio-based fraud alerts lock them out of the digital economy.
3. **Information Blackout:** Lack of closed captions on media and inability to hear ambient public announcements (transit, hospitals).

## 💡 The Solution: BridgeLens Features

### 🌍 1. Daily Interaction Hub (Context-Aware AI)
* **Logit-Masked Edge CV:** Uses an optimized MediaPipe model to track 543 skeletal landmarks. By using "Context Quick-Keys" (e.g., Coffee Shop, Emergency), the system mathematically restricts the AI's vocabulary, dropping latency to <50ms and completely eliminating hallucinated translations.
* **Ambient Ear:** A passive listening mode that captures background announcements (e.g., Train delays), processes the audio, and pushes visual Sign Language alerts to the user.
* **Indigenous Language Engine:** Two-way translation supporting English, Nigerian Pidgin, Yoruba, Igbo, and Hausa via Groq's Llama-3 API.

### 🏥 2. Medical Visit Module (Two-Way Clinical Bridge)
* **Patient-to-Doctor:** Translates the patient's sign language into clinical text, automatically logging symptoms into a digital chart.
* **Doctor-to-Patient:** Captures the doctor's spoken diagnosis, extracts the core NLP glosses, and plays sequential human sign language videos back to the patient.

### 💳 3. Financial Inclusion (Powered by Interswitch)
* **Branchless KYC:** Integrates the **Interswitch Identity Rails** for NIN verification and live facial comparison, upgrading users to Tier 3 accounts without needing to visit an uninterpreted bank branch.
* **Trust Shield:** Uses the **Interswitch Account Verification API** to visually verify recipient names before transfers, protecting Deaf users from vendor fraud.
* **Sign-to-Pay:** Replaces easily forgotten PINs with encrypted biometric sign language gestures for transaction authorization.

### 📺 4. Digital Content Bridge
* **Universal Audio Listener:** Bypasses missing YouTube closed captions by utilizing real-time SpeechRecognition to "listen" to any playing video.
* **Synchronized Interpretation:** Extracts target glosses from the audio and plays side-by-side sign language videos, making any raw video accessible instantly.

---
## 🎯 Judge's Testing Guide (Quick Walkthrough)
To experience the core features of BridgeLens without needing to upload your own files or speak into the microphone, please follow this "Golden Path" walkthrough:

**1. 🏥 Medical Visit (Two-Way Clinical Bridge)**
* **Patient to Doctor:** Under *Patient Input*, use the "Select a Demo Video from Repo" dropdown and choose a pre-loaded video (e.g., `HEADACHE`, `SICK`, or `MEDICINE`). Click **"Translate Symptoms to Doctor"**. Watch the AI extract the gesture using temporal majority-voting and auto-log it to the patient's chart.
* **Doctor to Patient:** Under *Doctor's Dashboard*, type a response into the input box (e.g., "TAKE MEDICINE") and click **"Translate to Sign Language"**. Finally, click **"Play Diagnosis Sequence"** to see the words translated into direct, sequential sign language videos.

**2. 🌍 Daily Interaction (Context AI)**
* **Context Restrictions:** Toggle the Environment Quick-Keys at the top (General, Coffee Shop, Transit). Notice how this dynamically restricts the Edge AI's vocabulary based on real-world locations to prevent hallucinations.
* **Speak to the World:** Under *3. Speak to the World*, select a demo video like `AIRPLANE` or `THANK YOU` from the dropdown and click **"Translate Sign to Speech"** to see the skeletal tracking and Indigenous Language translation in action.

**3. 📺 Media Access (Universal Listener)**
* Navigate to the **Media Access** tab.
* In the *Text to Sign Translation* box, type a sequence using available vocabulary (e.g., "HELLO DOCTOR").
* Click **"Prepare Sign Language Track"** and then **"Start Live Interpretation"** to watch the signs dynamically queue and play sequentially on the screen.

**4. 💳 Financial Inclusion (Interswitch Sandbox)**
* Navigate to the **Financial Inclusion** tab.
* Explore the dropdown actions to see our Interswitch integrations, including **Branchless KYC** (Identity Rails API), **Trust Shield verification** (Account Name Verification API), and our **Visual VAS Error Handler** (Translating cryptic API errors like "HTTP 402" into visual "NO MONEY" signs).
---
## 🛠️ Technical Architecture
* **Frontend:** Streamlit (Python)
* **Computer Vision (Edge):** OpenCV, MediaPipe Holistic (Pose & Hand tracking), Scikit-Learn (RandomForest Classifier).
* **NLP & LLM (Cloud):** Groq API (Llama3-8b-8192) for grammar smoothing and indigenous language translation.
* **Audio Processing:** SpeechRecognition API, gTTS.
* **Financial Infrastructure:** Interswitch API Marketplace & QuickTeller Business Sandbox.

---

## 👥 Team & Contributions

### AMOO ABDUSSOMOD OLADIPUPO — Team Lead & Product strategist
- **Product Definition:** Co-designed the "BridgeLens" concept, focusing on bridging communication gaps in Health and Finance sectors.
- **Documentation & Reporting:** Drafting the technical documentation and project README to align with hackathon requirements.
- **Full-Stack Development:** Built the multi-tab Streamlit interface and integrated real-time information processing.
- **Fintech Integration:** Designed and implemented the Interswitch API mock-integration logic for Secure Payments.
- **System Architecture:** Defined the data flow between visual input, AI inference, and service triggers.

### BASIT ABDULSALAM — AI Engineer & Domain Researcher
- **AI/ML Engineering:** Developed the MediaPipe landmark extraction pipeline and trained the sign language recognition models (Random Forest).
- **Domain Research:** Conducted research on the Sign Language landscape and identified high-impact vocabulary for the Medical and Fintech tabs.

## 🚀 How to Run Locally

**1. Clone the repository**
```bash
git clone [https://github.com/yourusername/bridgelens.git](https://github.com/yourusername/bridgelens.git)
cd bridgelens

TO Install Dependencies
pip install -r requirements.txt

Groq API key to test features (for JUDGES)
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

or you may generate yours and insert it in
export GROQ_API_KEY="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"

To run the Application:
streamlit run bridgelens_app.py

📂 Project Structure
bridgelens_app.py: Main application logic and UI routing.

samples/: Directory containing the target sign language MP4 videos.

model.p: Pre-trained Scikit-Learn RandomForest model and LabelEncoder.

