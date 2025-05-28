# Enhanced NDA Document Analysis Assistant

## Description

This project is an **Enhanced NDA Document Analysis Assistant** designed to help users analyze Non-Disclosure Agreement documents efficiently. It integrates with Google Drive to access NDA files and utilizes specific guideline documents to enhance the analysis process.

## Features

-   **Process Files**: Analyze NDA files provided by users.
-   **Google Drive Integration**: Connects to your Google Drive to list and view NDA documents.
-   **Guidelines Integration**: Learns from and applies guidelines for reviewing and amending documents.
-   **Intelligent Analysis**: Analyzes NDA documents and provides detailed feedback based on standard criteria and loaded guidelines.
-   **Comprehensive Reporting**: Delivers structured reports on violations, unclear clauses, suggested corrections, and completeness checks.

## Setup

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd Agents_NDA
    ```

2.  **Set up environment variables:**

    Create a `.env` file in the project root with the following variables:

    ```dotenv
    MODEL=gemini-2.0-flash # or other preferred model
    GOOGLE_CREDENTIALS_PATH=./path/to/your/google/credentials.json
    ```

3.  **Obtain Google Service Account Credentials:**

    Follow the instructions [here](link_to_google_credentials_docs) to create a Google Service Account and download the JSON credentials file. Place the file and update the `GOOGLE_CREDENTIALS_PATH` in your `.env` file.

4.  **Install dependencies:** (Assuming dependencies like `google-api-python-client`, `PyPDF2`, `python-dotenv`, `google-auth` are used)

    ```bash
    pip install -r requirements.txt
    ```
    (Note: A `requirements.txt` file is not included in the provided context. You may need to create one based on the imports in `agent.py`.)

## Usage

Run the agent script:

```bash
python agent.py
```

Interact with the agent using the following commands:

-   `List NDA files`: Shows available NDA documents from Google Drive.
-   `View file [number]`: Displays the content of a specific file from the list.
-   `Load guidelines`: Loads guideline documents from Google Drive for enhanced analysis.
-   `Show guidelines`: Displays a preview of the loaded guidelines.
-   `Analyze this document`: Initiates the analysis of the provided NDA content (either pasted or from a viewed file). **The agent will prompt you to specify the legal system (Thailand or Singapore) and confirm guideline usage before proceeding.**

## Analysis Output Format

The analysis results are structured into the following sections:

-   **Violations Identified**: Lists detected rule violations with original text, explanation, and reference (including page/paragraph if found).
-   **Unclear or Unreasonable Clauses**: Lists ambiguous or problematic clauses with original text, explanation, suggestion, and reference (including page/paragraph if found).
-   **Corrected Clauses**: Provides rewritten versions of problematic clauses.
-   **Completeness Check**: Verifies the presence of essential NDA clauses.
-   **Summary**: Provides a concise overview of the analysis findings.

## Contributing

(Add contributing guidelines here if applicable)

## License

(Add license information here if applicable)