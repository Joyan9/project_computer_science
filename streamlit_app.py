import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Sequence
from google.cloud import documentai
import os
import json
from PyPDF2 import PdfReader
import requests

# Set the default theme to wide
st.set_page_config(layout="wide")

# Convert the AttrDict to a regular dictionary
service_account_info = dict(st.secrets["gcp_service_account"])

# Write the service account info to a temporary JSON file
with open("service_account_key.json", "w") as f:
    json.dump(service_account_info, f)

# Set the environment variable for Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service_account_key.json"

PROJECT_ID = "536835879706"
LOCATION = "us"
PROCESSOR_ID = "d37e84f146ab793a"

# Title note with center alignment
st.markdown("""
    <h4 style="text-align: center;">*Web App Not Active - Archived</h4>
    """, unsafe_allow_html=True)

# Center-align the title of the Streamlit app
st.markdown('<h1 style="text-align: center;">Bank Statement Analyzer</h1>', unsafe_allow_html=True)

# Add some vertical space
st.write("\n" * 10)  # This will add 5 new lines as space
# Some introductory content
st.write("Welcome to the Bank Statement Analyzer. This tool will help you analyze your bank statements by providing you an interactive dataframe and a few visualisations based on the transactions. This web-app was created for academic purpose only and not for production level. Secondly, the app does not store any data on the servers.")

# Add some vertical space
st.write("\n" * 10)  # This will add 5 new lines as space
    
# Function to process the PDF online
@st.cache_data(show_spinner=False)
def online_process(
    project_id: str,
    location: str,
    processor_id: str,
    file_content: bytes,
    mime_type: str,
) -> documentai.Document:
    opts = {"api_endpoint": f"{location}-documentai.googleapis.com"}
    documentai_client = documentai.DocumentProcessorServiceClient(
        client_options=opts)
    resource_name = documentai_client.processor_path(
        project_id, location, processor_id)

    raw_document = documentai.RawDocument(
        content=file_content, mime_type=mime_type)
    request = documentai.ProcessRequest(
        name=resource_name, raw_document=raw_document)
    result = documentai_client.process_document(request=request)
    return result.document

# Function to get table data
def get_table_data(
    rows: Sequence[documentai.Document.Page.Table.TableRow], text: str
) -> List[List[str]]:
    if not rows:
        st.error("No table data found in the document. Please ensure the PDF contains valid tables.")
        return []
    
    all_values: List[List[str]] = []
    for row in rows:
        current_row_values: List[str] = []
        for cell in row.cells:
            current_row_values.append(
                text_anchor_to_text(cell.layout.text_anchor, text)
            )
        all_values.append(current_row_values)
    return all_values

def text_anchor_to_text(text_anchor: documentai.Document.TextAnchor, text: str) -> str:
    response = ""
    try:
        for segment in text_anchor.text_segments:
            try:
                start_index = int(segment.start_index)
                end_index = int(segment.end_index)
                # Check if indices are within the bounds of the text
                if start_index < 0 or end_index > len(text) or start_index >= end_index:
                    raise ValueError("Invalid text segment indices.")
                response += text[start_index:end_index]
            except (ValueError, AttributeError, IndexError) as e:
                # Handle specific exceptions for segment processing
                st.error(f"Error processing text segment: {e}")
                continue

        return response.strip().replace("\n", " ")
    except Exception as e:
        # Handle any other unexpected errors
        st.error(f"An unexpected error occurred: {e}")
        return "Error extracting text"


# Function to convert PDF to DataFrame

def convert_pdf_to_dataframe(
    file_content: bytes, project_id: str, location: str, processor_id: str, table_index: int
) -> pd.DataFrame:
    document = online_process(
        project_id=project_id,
        location=location,
        processor_id=processor_id,
        file_content=file_content,
        mime_type="application/pdf",
    )

    header_row_values = None
    all_body_row_values = []
    unique_header_values = set()
    for page in document.pages:
        for table in page.tables:
            if table.header_rows:
                header_values = get_table_data(
                    table.header_rows, document.text)
                unique_header_values.add(tuple(header_values[0]))

    unique_header_values = list(unique_header_values)
    
    #remove the empty strings
    unique_header_values = [sublist for sublist in unique_header_values if all(item != "" for item in sublist)]
    
    # remove lists with less than 4 items
    unique_header_values = [sublist for sublist in unique_header_values if len(sublist) > 4]
    
    st.session_state.selected_table = st.radio(
        "Please Select the table containing the transaction records:",
        unique_header_values,
        index=None
    )

    if st.session_state.selected_table:
        selected_table_index = unique_header_values.index(
            st.session_state.selected_table)
        st.write("You selected:", st.session_state.selected_table)

        selected_header_values = list(unique_header_values)[
            selected_table_index]

        include_row = False
        for page in document.pages:
            for table in page.tables:
                if table.header_rows:
                    header_values = get_table_data(
                        table.header_rows, document.text)
                    if tuple(header_values[0]) == selected_header_values:
                        header_row_values = header_values
                        include_row = True

                if include_row and len(header_row_values[0]) > 4:
                    body_row_values = get_table_data(
                        table.body_rows, document.text)
                    all_body_row_values.extend(body_row_values)

        # Ensure that all rows have the same number of columns as the header
        all_body_row_values = [row for row in all_body_row_values if len(
            row) == len(header_row_values[0])]

        df = pd.DataFrame(
            data=all_body_row_values,
            columns=pd.MultiIndex.from_arrays(header_row_values),
        )

        return df


# Store selected column mappings in session state
if "selected_columns" not in st.session_state:
    st.session_state.selected_columns = {}

# Categorize transactions based on description
@st.cache_data(show_spinner=False)
def categorize_transaction(description: str) -> str:
    categories = {
        'Food': ['restaurant', 'cafe', 'food', 'dining','club'],
        'Transport': ['uber', 'lyft', 'taxi', 'bus', 'train', 'flight'],
        'Shopping': ['amazon', 'store', 'shop', 'walmart', 'target'],
        'Utlity Bills': ['electricity', 'water', 'gas', 'internet'],
        'Entertainment': ['netflix', 'spotify', 'movie', 'cinema'],
        'Healthcare': ['pharmacy', 'doctor', 'hospital', 'clinic'],
        'Investment': ['stocks','clearance','fixed deposit','gold'],
        'Others': []
    }
    
    description = str(description).lower()
    for category, keywords in categories.items():
        if any(keyword in description for keyword in keywords):
            return category
    return 'Other'


def clean_and_convert_to_numeric(series: pd.Series) -> pd.Series:
    # Convert series to string to ensure string operations can be applied
    series = series.astype(str)
    # Remove extraneous characters
    series = series.str.replace(r'[^\d.-]', '', regex=True)
    # Replace multiple dots with a single dot (if necessary)
    series = series.str.replace(r'\.+', '.', regex=True)
    # Strip leading and trailing spaces
    series = series.str.strip()
    # Convert to numeric, coercing errors to NaN
    numeric_series = pd.to_numeric(series, errors='coerce')
    # Replace NaN values with 0 or another appropriate value
    numeric_series = numeric_series.fillna(0)
    return numeric_series

# Upload PDF file
uploaded_file = st.file_uploader(
    "Please Upload Your Bank Statement PDF file with less than 15 pages", type="pdf")

# URL of the sample bank statement PDF
pdf_url = "https://github.com/Joyan9/project_computer_science/raw/main/Sample%20Bank%20Statements/Wells%20Fargo%20Bank%20Sample%20Statement.pdf"

# Function to download the PDF content
@st.cache_data(show_spinner=False)
def download_pdf(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        st.error("Failed to download the PDF. Please check the URL.")
        return None


# Download the PDF
pdf_content = download_pdf(pdf_url)

if pdf_content:
    st.download_button(
        label="Download a Sample Bank Statement For Trial",
        data=pdf_content,
        file_name="Wells_Fargo_Bank_Sample_Statement.pdf",
        mime="application/pdf",
        use_container_width = True
    )
else:
    st.write("Unable to provide the sample bank statement. Please try again later.")

if uploaded_file is not None:
    # Create the temp directory if it doesn't exist
    temp_dir = "temp_pdf"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Save the uploaded file to the temp directory
    input_file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(input_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Open the PDF file and check the number of pages
    with open(input_file_path, "rb") as f:
        pdf_reader = PdfReader(f)
        num_pages = len(pdf_reader.pages)

    if num_pages > 15:
        st.error(
            "The uploaded PDF exceeds the 15-page limit. \nPlease upload a PDF with less than 15 pages.")
    else:
        result_df = convert_pdf_to_dataframe(
            uploaded_file.getvalue(),
            PROJECT_ID,
            LOCATION,
            PROCESSOR_ID,
            0,
        )

        if result_df is not None:
            headers = [col[0] for col in result_df.columns]

            st.write("Please map the required columns:")
            st.session_state.selected_columns['transaction_date'] = st.selectbox(
                "Select the column header matching with Transaction Date",
                headers,
                key='transaction_date',
                index=None
            )
            if st.session_state.selected_columns['transaction_date']:
                st.session_state.selected_columns['description'] = st.selectbox(
                    "Select the column header matching with Description",
                    headers,
                    key='description',
                    index=None
                )
                if st.session_state.selected_columns['description']:
                    st.session_state.selected_columns['debit'] = st.selectbox(
                        "Select the column header matching with Debit / Withdrawals",
                        headers,
                        key='debit',
                        index=None
                    )
                    if st.session_state.selected_columns['debit']:
                        st.session_state.selected_columns['credit'] = st.selectbox(
                            "Select the column header matching with Credit / Deposits",
                            headers,
                            key='credit',
                            index=None
                        )
                        if st.session_state.selected_columns['credit']:
                            st.session_state.selected_columns['account_balance'] = st.selectbox(
                                "Select the column header matching with Account Balance",
                                headers,
                                key='account_balance',
                                index=None
                            )

                            if st.session_state.selected_columns['account_balance']:
                                selected_columns = st.session_state.selected_columns
                                selected_column_values = list(
                                    selected_columns.values())

                                filtered_df = result_df[selected_column_values].copy(
                                )
                                filtered_df.columns = [
                                    key for key in selected_columns.keys()]
                                
                                
                                currency_symbol = st.text_input(
                                    "Enter the currency symbol")
                               
                                
                                #Cleaning begins
                                filtered_df.replace({'transaction_date': {"": np.nan}}, inplace=True)
                                filtered_df.replace({'account_balance': {"": np.nan}}, inplace=True)
                                filtered_df.replace({'description': {"": np.nan}}, inplace=True)

                                # Drop rows only if all columns are NaN
                                filtered_df.dropna(subset = ['description'], inplace=True)

                                # Categorize transactions
                                filtered_df['category'] = filtered_df['description'].apply(
                                    categorize_transaction)


                                # Replace None values with 0 for 'debit' and 'credit' columns
                                filtered_df.replace({'debit': {"": 0, None: 0, np.nan: 0},
                                                     'credit': {"": 0, None: 0, np.nan: 0}}, inplace=True)
                                
                                # Apply the function to the 'debit' and 'credit' columns
                                filtered_df['debit'] = clean_and_convert_to_numeric(filtered_df['debit'])
                                filtered_df['credit'] = clean_and_convert_to_numeric(filtered_df['credit'])
                                filtered_df['account_balance'] = clean_and_convert_to_numeric(filtered_df['account_balance'])
                                # Add another horizontal divider using HTML
                                st.markdown('<hr style="border:1px solid white">', unsafe_allow_html=True)
                                
                                # Add a section header for Data Visualization
                                st.markdown('<h2 style="text-align: center;">Interactive DataFrame</h2>', unsafe_allow_html=True)
                                st.write("This section will display your bank statement as an interactive dataframe")
                                edited_df = st.data_editor(filtered_df,
                                               column_config={
                                                   "debit": st.column_config.NumberColumn(
                                                       "Debit / Withdrawals",
                                                       format=f'{currency_symbol}%d'),
                                                   "credit": st.column_config.NumberColumn(
                                                       "Credit / Deposits",
                                                       format=f'{currency_symbol}%d'),
                                                   "account_balance": st.column_config.NumberColumn(
                                                       "Account Balance",
                                                       format=f'{currency_symbol}%d'),
                                                   "category": st.column_config.SelectboxColumn(
                                                       "Category",
                                                       width = "Large",
                                                       help="Please select appropriate category for transaction",
                                                       options=[
                                                                "Food",
                                                                "Transport",
                                                                "Shopping",
                                                                "Utlity Bills",
                                                                "Entertainment",
                                                                "Education",
                                                                "Investment",
                                                                "Others"
                                                            ],
                                                       required=True
                                               )
                                                }
                                               
                                                       )
                                
                                # Add another horizontal divider using HTML
                                st.markdown('<hr style="border:1px solid white">', unsafe_allow_html=True)
                                
                                # Add a section header for Data Visualization
                                st.markdown('<h2 style="text-align: center;">Data Visualisation</h2>', unsafe_allow_html=True)


                                # DATA VISUALISATION
                                # Chart 1: Debit vs Credit
                                # Total debit and credit
                                total_debit = filtered_df['debit'].sum()
                                total_credit = filtered_df['credit'].sum()
                                
                                # Define data and labels
                                labels = [f'Total Debit ({currency_symbol}{int(total_debit):,})',
                                          f'Total Credit ({currency_symbol}{int(total_credit):,})']
                                values = [total_debit, total_credit]
                                
                                # Define colors
                                colors = ['#ff9999', '#66b3ff']
                                
                                # Create the doughnut chart
                                fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='percent', hoverinfo='label+percent',
                             marker=dict(colors=colors, line=dict(color='white', width=2)),
                             hole=0.4)])
                                
                                fig.update_layout(title='#1 Income to Spend Ratio', title_font_size=18, title_font_weight='bold', title_x=0.0)
                                
                                # Show the pie chart in Streamlit
                                st.plotly_chart(fig, use_container_width=True)
        
                                
                                # Chart 2: Spend by Category
                                # Grouping by category and summing up the debit values
                                category_debit_sum = edited_df.groupby('category')['debit'].sum().reset_index()
                                
                                # Create the bar chart
                                fig = px.bar(category_debit_sum, x='debit', y='category', orientation='h', 
                                             labels={'debit': 'Total Debit', 'category': 'Category'},
                                             text=category_debit_sum['debit'].astype(int).apply(lambda x: f"{currency_symbol}{x:,.0f}"))
                                
                                # Customize the layout
                                fig.update_layout(title='#2 Total Debit by Category', xaxis_title="Amount Spent", yaxis_title="Category")
                                
                                # Make the text bolder
                                fig.update_traces(textfont=dict(weight='bold'))
                                
                                # Display the chart
                                st.plotly_chart(fig, use_container_width=True)
                                

        
