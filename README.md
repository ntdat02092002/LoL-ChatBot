---
title: LoL Patch Update Chat-bot
emoji: 🏆
colorFrom: green
colorTo: indigo
sdk: streamlit
sdk_version: "1.39.0"
app_file: chatbot.py
pinned: false
python_version: 3.11
---

# League of Legends Patch Update ChatBot

An interactive chatbot that provides users with real-time access to the latest patch updates for League of Legends.

🚀 **Live Demo:** [League Patch ChatBot](https://ntdat02092002-lol-chatbot.hf.space)

## 🛠 Tech Stack

- **RAG Chatbot:** Google API for LLMs & text embedding, LangChain
- **Vector Database (Patch Info Storage):** Pinecone
- **Automated Data Flow (Crawling & Storing Patch Data):** Prefect, BeautifulSoup
- **Database (Logging Full Chatbot Interactions for Future Optimization):** Weights & Biases (WandB)
- **User Interface:** Streamlit

## 🚀 Getting Started

### For Regular Users

Simply visit the [Live Demo](https://ntdat02092002-lol-chatbot.hf.space) and start chatting!

### For Developers (Build Your Own)

#### 1️⃣ Clone the Repository & Install Dependencies

```bash
# Clone the repo
git clone https://github.com/ntdat02092002/LoL-ChatBot
cd LoL-ChatBot

# Install dependencies
pip install -r requirements.txt
```

#### 2️⃣ Set Up Required Accounts & API Keys

You will need API keys from the following services:

- [Google AI Studio](https://aistudio.google.com/)
- [Prefect](https://www.prefect.io/)
- [Weights & Biases (WandB)](https://wandb.ai/site)
- [Pinecone](https://www.pinecone.io/)

⚠️ **Note:** All platforms offer free tiers (subject to change in the future). Visit their official sites for the latest details.

#### 3️⃣ Configure Environment Variables

Create a `.env` file or export the following environment variables:

```bash
WANDB_API_KEY=<your_wandb_api_key>
PINECONE_API_KEY=<your_pinecone_api_key>
GOOGLE_API_KEY=<your_google_api_key>
PINECONE_INDEX_NAME=<your_pinecone_index_name>
PREFECT_API_KEY=<your_prefect_api_key>
PREFECT_API_URL=<your_prefect_api_url>
```

## 🔄 Automate Data Flow (Daily Patch Updates)

### Option 1: Run Locally

For setting up and deploying on your local machine, please refer to the official guide [here](https://docs.prefect.io/v3/deploy/run-flows-in-local-processes) using `pinecone_data_flow.py` script.

### Option 2 (Recommended): Deploy to Prefect Cloud

```bash
python prefect_deployment.py
prefect deployment run 'pinecone-flow/lolchatbot-data-flow'
```

## 🤖 Start the Chatbot

```bash
streamlit run chatbot.py
```

Visit the provided `localhost` link in your browser to access the chatbot UI.

## 🔧 Customization (Use Different AI Models)

For those who want to experiment with different LLMs or text embedding models (e.g., OpenAI, Claude, etc.), check out [LangChain Documentation](https://python.langchain.com/) for model integrations.

To modify the chatbot:

- Edit `config.json` for model settings
- Update `main.py` and `pinecone_data_flow.py` accordingly

---

## 🎯 Features

✅ Automatically crawls detailed patch notes from the official game site to ensure up-to-date information.

✅ RAG-based retrieval model using LangChain with Pinecone as the vector database.

✅ Implements **self-query retriever** for more accurate and context-aware responses.

✅ Stores complete user interaction history **under the hood** for analysis and continuous improvement.

## 🚀 Roadmap

🔹 Optimize prompt templates for improved response quality

🔹 Implement memory & follow-up conversation capabilities

🔹 Expand chatbot scope beyond latest patch updates (include patch history & general game information)

---

💡 **Contributions & Feedback:** Feel free to open issues or PRs to improve this project!


