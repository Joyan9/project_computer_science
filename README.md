# Bank Statement Analysis Project

This project is developed as part of the Computer Science course. It involves analyzing bank statements using Python, Streamlit, and Google Cloud's Document AI. The application processes PDF bank statements, extracts relevant financial data, and provides insights through visualizations.

## Features

- Upload and process PDF bank statements
- Extract transaction details using Google Cloud Document AI
- Clean and analyze transaction data
- Identify and remove discrepancies in account balances
- Visualize income and expenditure through interactive charts

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Google Cloud Document AI
- PyPDF2

## Setup and Installation

### Prerequisites

- Python 3.9+
- Google Cloud account with Document AI enabled
- Service account key for Google Cloud

### Installation

1. **Clone the Repository**

    ```bash
    git clone https://github.com/yourusername/bank-statement-analysis.git
    cd bank-statement-analysis
    ```

2. **Create and Activate Virtual Environment**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4. **Configure Google Cloud Credentials**

    Add your Google Cloud service account key to Streamlit secrets. Create a file named `secrets.toml` in the `.streamlit` directory with the following content:

    ```toml
    [gcp_service_account]
    type = "service_account"
    project_id = "your_project_id"
    private_key_id = "your_private_key_id"
    private_key = "your_private_key"
    client_email = "your_client_email"
    client_id = "your_client_id"
    auth_uri = "https://accounts.google.com/o/oauth2/auth"
    token_uri = "https://oauth2.googleapis.com/token"
    auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
    client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/your_client_email"
    ```

    Make sure to add `secrets.toml` to `.gitignore` to prevent it from being uploaded to GitHub.

    ```plaintext
    # .gitignore
    secrets.toml
    ```

5. **Run the Application**

    ```bash
    streamlit run streamlit_app.py
    ```

## Usage

1. Upload a PDF bank statement using the file uploader in the Streamlit app.
2. The application will process the PDF and extract transaction data.
3. View the extracted transactions and identified discrepancies.
4. Explore the visualizations to gain insights into your income and expenditure patterns.

## Project Structure

- `streamlit_app.py`: Main application script
- `requirements.txt`: List of dependencies
- `.streamlit/secrets.toml`: Google Cloud credentials (not included in the repository)

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License.

## Acknowledgements

- This project uses Google Cloud Document AI for data extraction.
- Developed as part of the Computer Science course project.

## Contact

For any queries or feedback, please contact joyan-sharukh.bhathena@iu-study.org
