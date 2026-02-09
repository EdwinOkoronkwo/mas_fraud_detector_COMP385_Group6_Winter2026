🛡️ Multi-Agent Fraud Detection System (MAS-Fraud)

An advanced fraud investigation pipeline that leverages a Multi-Agent System (MAS) to analyze financial transactions. The system integrates Supervised Learning (RNN/LR) and Unsupervised Anomaly Detection (DBSCAN) through a Dynamic Logic Gatekeeper.
🚀 Key Features

    Adaptive Ensemble Logic: Automatically weights model influence based on real-time accuracy. If a model’s accuracy drops to 50% (random guess), its weight is nullified to protect the bottom line.

    Synergistic Conviction: Uses weighted RMS (Root Mean Square) aggregation to reward "sharp" signals over dull averages, ensuring the AI captures the highest fraud savings.

    Automated Gatekeeper Rules:

        LowValueShield: Prevents high-friction blocks on minor transactions.

        ExtremeAnomaly: High-conviction "double-down" when multiple models detect high-risk patterns.

        HighValueConsensus: Forces a re-evaluation if multiple models suspect fraud on large transactions.

    Financial ROI Auditing: Generates detailed reports and visualizations correlating detection precision directly to dollar-value savings.

🏗️ System Architecture

The project is structured using the Facade Pattern, where the FraudInferencePipeline orchestrates specialized components:

    TaskArchitect: Converts raw DB rows into structured investigation tasks.

    AgentOrchestrator: Manages the RAG-enabled agent team to extract scores.

    LogicGatekeeper: The "Brain" that applies accuracy-weighted math and safety rules.

    ReportGenerator: Produces performance charts (performance_report.png) and risk heatmaps (risk_heatmap.png).

📈 Understanding the "Value-Weighted" Logic

Traditional ensembles often average scores, which allows a poor model (like 50% accurate LR) to drag down the AI's performance. Our system uses a Utility Function:
Utility=max(0,0.5Accuracy−0.5​)

This ensures that:

    50% Accuracy = 0.0 Weight (The "Coin Flip" Filter)

    75% Accuracy = 0.5 Weight

    100% Accuracy = 1.0 Weight

Consequently, on the Net Financial Savings chart, a model that guesses randomly will show $0.00 in savings, while the AI Ensemble—leveraging the best of RNN and DB—takes the lead.
🛠️ Installation & Setup
1. Prerequisites

Ensure you have uv installed. It is the recommended runner for this project's pyproject.toml structure.
Bash

# If you don't have uv yet
powershell -c "ir | iex" # Windows
curl -LsSf https://astral-sh.uv.io/install.sh | sh # macOS/Linux

2. Clone the Repository
Bash

git clone https://github.com/EdwinOkoronkwo/mas_fraud_detector.git
cd mas_fraud_detector

3. Initialize Environment & Sync Dependencies

The pyproject.toml contains specific requirements for Python >=3.12. UV will automatically create a virtual environment and install everything (including autogen-ext, torch, and scikit-learn) in one step:
Bash

uv sync

4. Configure Environment Variables

Create a .env file in the root directory to store your API credentials:
Ini, TOML

OPENAI_API_KEY=your_key_here
# Optional: MISTRAL_API_KEY=your_key_here

5. Execution

To run the batch inference and generate the Accuracy-to-Savings reports:
Bash

uv run python -m mas_fraud_detector.inference.run_inference

📦 Dependency Highlights (from pyproject.toml)

dependencies = [
    "autogen-agentchat",
    "autogen-core",
    "autogen-ext",
    "pandas",            # Added for DataHandler and Pipeline loops
    "numpy",             # Added for RMS math in LogicGatekeeper
    "ipykernel",
    "python_dotenv",
    "tiktoken",
    "ollama",
    "openai",
    "mistralai",
    "json-schema-to-pydantic",
    "langchain-community",
    "arxiv",
    "wikipedia",
    "streamlit",
    "docker",
    "autogen-ext[docker]",
    "scikit-learn",
    "imbalanced-learn",
    "seaborn",
    "matplotlib",
    "autogen-ext[mistral]",
    "xgboost",
    "minisom",
    "torch",
    "langchain",
    "langchain-openai",
    "chroma",
    "chromadb",
    "tabulate"
]

🧪 Post-Installation Verification

Once installed, verify the setup by checking if the visualization engine is ready:
Bash

uv run python -c "import matplotlib; print('Graphics Engine Ready')"

📊 Sample Output

After a batch run, the system generates a Risk Topology Heatmap:

    Safe Zone (Green): Low risk, automated pass.

    Ambiguity Zone (Yellow): Requires consensus or manual review.

    Danger Zone (Red): High-conviction fraud detected by the AI Ensemble.

📄 License

Distributed under the MIT License. See LICENSE for more information.