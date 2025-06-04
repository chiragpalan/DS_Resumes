
import os
import win32com.client

# Folder containing the Excel files
folder_path = r"C:\path\to\your\folder"

# Launch Excel
excel = win32com.client.Dispatch("Excel.Application")
excel.Visible = False  # Set to True to watch it work

for file_name in os.listdir(folder_path):
    if file_name.endswith(".xlsm"):  # Ensures macro-enabled files only
        file_path = os.path.join(folder_path, file_name)
        wb = excel.Workbooks.Open(file_path)

        try:
            ws_source = wb.Sheets("NEW")
            ws_dest = wb.Sheets("Sheet2")

            # Copy B11:R18 (i.e., rows 11-18, cols B to R)
            data = ws_source.Range("B11:R18").Value
            ws_dest.Range("A1:Q8").Value = data  # A1:Q8 is 8 rows × 17 columns

            # Call the macro
            excel.Application.Run("chirag_macro")  # Assumes macro is workbook-level

            # Save and close
            wb.Save()
            wb.Close()

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

excel.Quit()
print("All files updated and macro executed.")
