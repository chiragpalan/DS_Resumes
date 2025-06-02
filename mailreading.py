import extract_msg
from bs4 import BeautifulSoup
import pandas as pd

def extract_tables_from_msg(msg_path):
    """
    Extracts all HTML tables from a .msg file and returns them as a list of pandas DataFrames.

    Parameters:
        msg_path (str): The full path to the .msg file.

    Returns:
        List[pd.DataFrame]: A list of DataFrames, each representing a table found in the message.
    """
    # Load the Outlook .msg file
    msg = extract_msg.Message(msg_path)

    # Try to get the HTML body first, fallback to plain text if not available
    html_content = msg.htmlBody or msg.body

    if not html_content:
        print(f"No HTML or plain body found in {msg_path}")
        return []

    # Try parsing the message body using lxml, fall back to html.parser if lxml is not available
    try:
        soup = BeautifulSoup(html_content, 'lxml')
    except Exception:
        soup = BeautifulSoup(html_content, 'html.parser')

    # Find all <table> elements in the HTML
    tables = soup.find_all("table")

    dataframes = []

    # Loop through each table found
    for table in tables:
        rows = table.find_all("tr")  # Extract all table rows
        table_data = []

        # Loop through each row and extract cell contents
        for row in rows:
            cols = row.find_all(["td", "th"])  # Cells can be <td> or <th>
            row_data = [cell.get_text(strip=True) for cell in cols]  # Clean and strip cell text
            table_data.append(row_data)

        # Convert to DataFrame only if table has data
        if table_data:
            # If the first row looks like a header (all non-empty), use it as column names
            if all(cell != '' for cell in table_data[0]):
                df = pd.DataFrame(table_data[1:], columns=table_data[0])
            else:
                df = pd.DataFrame(table_data)

            dataframes.append(df)

    return dataframes
