# Hierarchical Multi-Agent System with Local LLMs

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project is a proof-of-concept implementation of a hierarchical multi-agent architecture designed to orchestrate Large Language Models (LLMs) for complex task execution. It moves beyond a single, monolithic agent to a modular system where an **Orchestrator Agent** delegates tasks to a pool of specialized **Expert Agents**.

The core principle is to leverage the powerful function-calling (or "tool-calling") capabilities of modern LLMs for routing, while smaller, specialized agents handle the actual task execution. This significantly reduces the context window for any single LLM call, improving latency, accuracy, and resource efficiency.

## Core Concept

The architecture is built on a two-tier decision-making process:

1.  **Orchestrator Agent (Main-Agent):** This is the user-facing entry point. Its sole responsibility is to analyze an incoming query and determine which specialized agent is best suited to handle it. It does not perform the task itself but instead calls the appropriate expert agent as a "tool."

2.  **Expert Agents (Sub-Agents):** Each expert is a self-contained agent focused on a narrow domain. It has its own system prompt and a small, highly relevant set of tools (e.g., a specific RAG pipeline, an API call). It receives the task from the orchestrator, executes it using its tools, and returns a structured result.

This approach combines the probabilistic nature of LLMs (for intelligent routing) with the deterministic execution of code (for reliable task completion), creating a robust and scalable system.

## Technology Stack

*   **AI Framework:** **PydanticAI** is used to create strongly-typed, structured interactions with the LLM. It allows agents and tools to be defined as Python functions with validated inputs and outputs, which is critical for the reliability of the system.
*   **LLM Inference:** The system runs entirely locally using the **`llama.cpp` server**. This provides an OpenAI-compatible API endpoint, allowing for high-performance inference on local hardware with full data privacy.
*   **Language Model:** The architecture has been developed and tested with **Qwen3-30B-A3B**, a powerful Mixture-of-Experts (MoE) model with excellent tool-calling capabilities. Any modern LLM with robust function/tool-calling support in GGUF format should work.

---

## 🚀 Setup and Usage

### Prerequisites

*   Python 3.11+ and Pip
*   Git
*   A C++ compiler (for building `llama.cpp`)
*   A powerful GPU (NVIDIA recommended) with sufficient VRAM to run the model.

### 1. Clone the Repository

```bash
git clone https://github.com/CeeArEx/oth-chatbot.git
cd oth-chatbot
```

### 2. Set Up the Environment

It is highly recommended to use a virtual environment.

```bash
# Create and activate a virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Download the LLM

1.  Download a GGUF-quantized version of the **[Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B)** model (or a similar tool-calling model) from a source like Hugging Face.
2.  Create a `models` directory in the project root and place the `.gguf` file inside.

### 4. Start the llama.cpp server
The Github repo of [llama.cpp](https://github.com/ggml-org/llama.cpp) describes in detail how to start the llama.cpp server.

The server will load the model and wait for requests. Leave this terminal running.

### 5. Start the Chatbot CLI

First configure the .env file with the ip of the server where you started the llm.

```txt
LLM_NAME="Qwen3-30B-A3B"
LLM_URL="xxx.xxx.xxx.xxx"
LLM_PORT="xxxx"
```

After that, activate the virtual environment and run the main Python script.

```bash
# In a new terminal
source venv/bin/activate
python main.py
```

The application will connect to the local server and you can begin interacting with the chatbot via the command line.

---

### ⚠️ **Important Project Focus**

Beware that the current implementation of the **Expert Agents and their tools are highly specialized**. They are hard-coded to retrieve and process information from the web pages of the **[OTH Amberg-Weiden](https://www.oth-aw.de/)** .

To adapt this project for a different purpose, you will need to:
1.  Redefine the Expert Agents in the `agents/` directory.
2.  Implement new, relevant tools in the `tools/` directory.
3.  Update the Orchestrator Agent's configuration to recognize and route to your new agents.

## 📄 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.
