# 🌍 Travel Itinerary Planner

**AI-powered day trip planner** that generates personalized itineraries based on your chosen **city** and **interests** — all in seconds.  
Built using **LangGraph**, **LangChain**, **Gradio**, and **Groq Llama 3.3 70B** for fast, intelligent trip suggestions.

--- 

### 🖼️ Preview
<img width="1756" height="792" alt="image" src="https://github.com/user-attachments/assets/05e5a645-059b-4ad9-8160-ede9bf14c5e8" />

---

## ✨ Features
- 🏙️ Generates full-day trip itineraries for any city  
- 💬 Understands user interests (e.g., *nature, food, culture, history*)  
- ⚡ Powered by **Groq Llama 3.3 70B** for lightning-fast results  
- 🎨 Beautiful Gradio UI with a custom orange-green theme  
- ☁️ Deployable on **Hugging Face Spaces** or any cloud platform  

---

## 🧩 Tech Stack

| Category | Technology |
|-----------|-------------|
| Frontend | [Gradio](https://gradio.app/) |
| Backend | [LangGraph](https://python.langchain.com/docs/langgraph), [LangChain](https://www.langchain.com/) |
| LLM | [Groq Llama 3.3 70B](https://groq.com/) |
| Language | Python 3.10 |
| Hosting | Hugging Face Spaces / Local / Cloud |

---

## 🚀 Demo
👉 **[Try it on Hugging Face](#)**  
*([Link](https://huggingface.co/spaces/anshu2025/Travel_itinenary_planner)!)*

---

## ⚙️ Setup & Installation

### 1️⃣ Clone this repo
```bash
git clone https://github.com/yourusername/travel-itinerary-planner.git
cd travel-itinerary-planner
```

### 2️⃣ Create a virtual environment
```
python -m venv venv
source venv/bin/activate   # for macOS/Linux
venv\Scripts\activate      # for Windows
```

### 3️⃣ Install dependencies
```
pip install -r requirements.txt
```

### 4️⃣ Add your environment variable
```
Create a .env file in the root directory:
GROQ_API_KEY=your_actual_api_key_here
Make sure .env is added to .gitignore so it’s never uploaded to GitHub:
.env
```
---

### 📦 Requirements
- gradio
- langchain
- langchain-core
- langchain-community
- langchain-groq
- langgraph
- python-dotenv
and a runtime.txt:
- python-3.10

### 🧑‍💻 Author
Anshu Priya
(AIML Engineer)
📍 Jaipur, Rajasthan

🪪 License

This project is licensed under the MIT License — you’re free to modify and distribute it with attribution.
