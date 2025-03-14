# Clinical AI Advisor using RAG and LLM

This repository presents a Clinical AI Advisor that integrates Retrieval-Augmented Generation (RAG) with Large Language Models (LLM) to deliver intelligent insights and recommendations for clinical decision-making. By combining RAG's efficient information retrieval capabilities with the contextual understanding of LLMs, this system aids healthcare professionals in making informed decisions.

![image](https://github.com/user-attachments/assets/cbdc2107-79d3-45ea-8fcf-15928a5565a9)


## Table of Contents

- [About](#about)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## About

The Clinical AI Advisor leverages advanced AI techniques to process and analyze medical data, providing real-time, context-aware responses. Its primary goal is to assist healthcare professionals by offering accurate information retrieval and generation capabilities, thereby improving patient outcomes and supporting clinical workflows.

## Features

* **Clinical Query Processing:** Handles and processes clinical questions.
* **Retrieval-Augmented Generation (RAG):** Integrates external knowledge retrieval with LLM responses.
* **Large Language Model (LLM) Integration:** Utilizes LLMs for generating informed responses.
* **User Interface:** Provides a web-based interface for easy interaction.

## Technologies Used

- **Retrieval-Augmented Generation (RAG)**: Combines retrieval and generation mechanisms for efficient information processing.
- **Large Language Models (LLMs)**: Advanced models like Mistral AI for natural language understanding and generation.
- **TensorFlow and PyTorch**: Frameworks used for implementing and training AI models.
- **Medical Data Sources**: Integration with authoritative medical databases for data retrieval.

## Installation

To set up the Clinical AI Advisor locally, follow these steps:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Saifulislamsayem19/Clinical-AI-Advisor-using-RAG-and-LLM.git
   cd Clinical-AI-Advisor-using-RAG-and-LLM
2.  **Create a Virtual Environment (optional but recommended):**

    ```bash
    python3 -m venv env
    source env/bin/activate  # On Windows, use 'env\Scripts\activate'
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables:**

    * Create a `.env` file in the root directory.
    * Add your LLM API key (e.g., `HF_TOKEN=your_huggingface_api_key`). Add any other necessary environment variables.

### Usage

1.  **Start the Application:**

    ```bash
    python llm_memory.py
    ```

2.  **Interact with the Advisor:**

    * Open your web browser and navigate to `http://127.0.0.1:5000/`.
    * Enter your clinical queries in the provided interface.

## Contributing

Contributions are welcome! Please follow these steps:

1.  **Fork the repository.**
2.  **Create a new branch** (`git checkout -b feature-name`).
3.  **Commit your changes** (`git commit -am 'Add new feature'`).
4.  **Push to the branch** (`git push origin feature-name`).
5.  **Create a new Pull Request.**

Please ensure your code adheres to existing coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
