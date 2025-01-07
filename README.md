# Learning Chatbot

An interactive chatbot that learns and adapts based on user interactions, built with LangChain, LangGraph, and Groq LLM.

## Features

- Interactive Streamlit-based UI with statistics dashboard
- Continuous learning from user interactions
- Fact validation system
- Preference management (tone, verbosity, formality)
- Real-time interaction statistics
- Vector-based knowledge storage using ChromaDB
- Docker containerization with health checks
- Automated testing suite

## Prerequisites

- Docker and Docker Compose
- Python 3.11 or higher (if running locally)
- Groq API key (already configured in .env)

## Installation

### Using Docker (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/antoniofullstack/chatbot
cd chatbot
```

2. Build and start the containers:
```bash
docker compose up --build
```

The application will be available at `http://localhost:8501`

### Local Installation

1. Create a virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/MacOS
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start ChromaDB:
```bash
docker-compose up chromadb -d
```

4. Run the application:
```bash
streamlit run src/app.py
```

## Project Structure

```
chatbot/
├── src/
│   ├── app.py          # Streamlit UI
│   ├── chatbot.py      # Core chatbot logic
│   └── config.py       # Configuration settings
├── data/
│   └── chromadb/       # Vector database storage
├── tests/              # Test suite
├── docker-compose.yml  # Container orchestration
├── docker-compose.test.yml  # Test container setup
├── Dockerfile         # Python environment setup
├── requirements.txt   # Python dependencies
├── requirements-test.txt  # Test dependencies
└── .env              # Environment variables
```

## Usage Examples

1. Starting a Conversation
   - Open your browser and navigate to `http://localhost:8501`
   - Type your message in the chat input field
   - The chatbot will respond based on its knowledge and learning capabilities
   - Monitor interaction statistics in the sidebar

2. Managing Preferences
   ```
   User: "I prefer formal communication"
   Chatbot: *adapts communication style to formal*
   
   User: "I like detailed explanations"
   Chatbot: *adjusts verbosity level*
   ```

3. Teaching and Querying
   ```
   User: "The capital of France is Paris"
   Chatbot: *validates and stores this fact*
   
   User: "What do you know about France?"
   Chatbot: *retrieves and presents stored information*
   ```

4. Viewing Statistics
   - Check the sidebar for:
     - Total messages exchanged
     - Facts learned
     - Questions asked
     - Preferences set

## Configuration

The application uses several environment variables:
- `GROQ_API_KEY`: API key for Groq LLM (pre-configured)
- ChromaDB settings are managed through Docker Compose

## Architecture

1. User Interface (Streamlit)
   - Handles user interactions
   - Displays chat history
   - Manages session state

2. Chatbot Core (LangChain + LangGraph)
   - Process input using LangGraph workflow
   - Validates facts using Groq LLM
   - Stores validated information in ChromaDB
   - Generates contextual responses

3. Vector Storage (ChromaDB)
   - Stores embeddings of learned information
   - Enables semantic search and retrieval
   - Persists data across sessions

## Testing

Run the test suite using:
```bash
# Using Docker
docker-compose -f docker-compose.test.yml up --build

# Local testing
pip install -r requirements-test.txt
./run_tests.sh
```

The test suite covers:
- Core chatbot functionality
- Fact validation
- Preference management
- Vector database operations

## Troubleshooting

1. If the application fails to start:
   - Check if Docker is running
   - Verify that ports 8501 and 8000 are available
   - Ensure the .env file contains the correct API key

2. If ChromaDB connection fails:
   - Restart the ChromaDB container:
     ```bash
     docker-compose restart chromadb
     ```

3. If changes aren't reflected:
   - Rebuild the containers:
     ```bash
     docker-compose down
     docker-compose up --build
     ```

## Development

To extend the chatbot's capabilities:

1. Add new LangGraph nodes in `src/chatbot.py`
2. Modify the conversation flow in `setup_graph()`
3. Update the Streamlit UI in `src/app.py`
4. Add new environment variables to `.env` and `docker-compose.yml`

## License
The MIT License (MIT)

Copyright (c) 2022 Antonio Cavedoni

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
