# 📚 AI Knowledge Agent - Sách & Báo Chí

An intelligent document research assistant and intellectual dialogue companion built with **Streamlit**, **ChromaDB**, and multi-model LLMs (**DeepSeek** & **Google Gemini**).

---

## 🌟 Key Features

1. **🔍 In-Depth Q&A (RAG)**: Ask precise conceptual and academic questions across your library with exact page citations and similarity scoring.
2. **⚖️ Cross-Perspective Debate**: Pit two authors or philosophical schools against each other (e.g. Karl Marx vs. Adam Smith, Stoicism vs. Buddhism) with balanced evidence synthesis.
3. **🧠 Insights & Mindmaps**: Extract core principles, practical life lessons, and generate visual Mermaid mindmaps.
4. **📤 Cloud & Local Ingestion**: Upload books directly through the web UI or scan local folders (supports `.pdf`, `.docx`, `.epub`).
5. **🤖 Dual LLM Integration**:
   - **DeepSeek**: `deepseek-chat`, `deepseek-reasoner`
   - **Google Gemini**: `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-1.5-flash`

---

## 🚀 Quick Start (Local)

### 1. Clone & Setup Virtual Environment
```bash
git clone https://github.com/Kiris-02/knowledge-agent.git
cd knowledge-agent
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment (Optional)
Create a `.env` file or enter keys directly in the sidebar UI:
```env
DEEPSEEK_API_KEY=your_deepseek_key
GEMINI_API_KEY=your_gemini_key
```

### 3. Launch App
```bash
streamlit run app.py
```
Or double-click `run_app.bat` on Windows.

---

## ☁️ Cloud Deployment

### Option A: Streamlit Community Cloud (Recommended & 100% Free)
1. Go to [share.streamlit.io](https://share.streamlit.io/).
2. Click **New app**.
3. Select Repository: `Kiris-02/knowledge-agent`.
4. Branch: `main`, Main file path: `app.py`.
5. Under **Advanced Settings**, add your Secrets (`DEEPSEEK_API_KEY`, `GEMINI_API_KEY`).
6. Click **Deploy!**

### Option B: Render Web Service
1. Go to [Render Dashboard](https://dashboard.render.com/) -> **New Web Service**.
2. Connect `Kiris-02/knowledge-agent`.
3. Settings:
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. Add Environment Variables: `DEEPSEEK_API_KEY`, `GEMINI_API_KEY`.
5. Click **Create Web Service**.
