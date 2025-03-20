# Clinical AI Advisor using RAG and LLM

The **Clinical AI Advisor** is an advanced artificial intelligence system that leverages the combination of **Retrieval-Augmented Generation (RAG)** and **Large Language Models (LLM)** to provide reliable and efficient clinical decision support. This project aims to enhance medical practitioners' capabilities by offering evidence-based recommendations, diagnostic insights, and more.

![image](https://github.com/user-attachments/assets/42411511-d3c6-481b-8779-d829d1898053)

## Features

- **AI-powered Clinical Guidance**: Provides context-aware and accurate medical advice.
- **Retrieval-Augmented Generation**: Combines information retrieval with generation capabilities to offer precise insights from a knowledge base.
- **Integration with LLM**: Utilizes large language models for understanding and generating human-like responses to medical queries.
- **Interactive User Interface**: Simple and intuitive interface for seamless interaction with the AI system.
- **Scalable**: Built with scalability in mind to handle various medical domains and use cases.

## Installation

To install and run the Clinical AI Advisor locally, follow these steps:

### Prerequisites
Ensure that the following software is installed on your system:

- Python 3.7+
- Pip (Python package manager)

### Steps to Install

1. Clone the repository:
   ```bash
   git clone https://github.com/Saifulislamsayem19/Clinical-AI-Advisor-using-RAG-and-LLM.git
2.  Navigate to the project directory:

    ```bash
    cd Clinical-AI-Advisor-using-RAG-and-LLM
    ```

3.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4.  Configure the environment (if needed):

    Follow the configuration instructions in the documentation for setting up the necessary environment variables or configurations.

5.  Run the application:

    ```bash
    python app.py
    ```

### Docker Setup (Optional)

You can also run the application using Docker for easier setup.

1.  Build the Docker image:

    ```bash
    docker build -t clinical-ai-advisor .
    ```

2.  Run the Docker container:

    ```bash
    docker run -p 5000:5000 clinical-ai-advisor
    ```

## Usage

Once the application is running, you can access the Clinical AI Advisor by visiting the following URL in your browser:

http://localhost:5000


The interface allows you to ask clinical-related questions, and the AI will generate responses based on the input and the knowledge it retrieves.

## Technologies Used

-   **Python**: The core programming language for the project.
-   **Hugging Face Transformers**: To access and utilize pre-trained large language models (LLMs).
-   **FAISS**: A library for efficient similarity search in large datasets, used to implement RAG.
-   **Flask**: The web framework used for building the web application.
-   **Docker**: For containerizing the application to simplify deployment and testing.

## Contributing

We welcome contributions from the community! If you would like to contribute to this project, please follow these steps:

1.  Fork the repository.
2.  Create a new branch:

    ```bash
    git checkout -b feature-branch
    ```

3.  Make your changes.
4.  Commit your changes:

    ```bash
    git commit -m 'Add new feature'
    ```

5.  Push to the branch:

    ```bash
    git push origin feature-branch
    ```

6.  Open a pull request to the `main` branch.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
