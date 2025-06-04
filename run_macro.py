import os
import win32com.client

# Folder containing your .xlsx files
folder_path = r"C:\path\to\your\folder"

# Launch Excel
excel = win32com.client.Dispatch("Excel.Application")
excel.Visible = False  # Set to True if you want to see it happen

# Ensure PERSONAL.XLSB is open to access the macro
# This assumes PERSONAL.XLSB opens automatically with Excel (usually does if macro is saved there)

for file_name in os.listdir(folder_path):
    if file_name.endswith(".xlsx"):
        file_path = os.path.join(folder_path, file_name)
        wb = excel.Workbooks.Open(file_path)

        try:
            ws_source = wb.Sheets("NEW")
            ws_dest = wb.Sheets("Sheet2")

            # Copy B11:R18 to A1:Q8
            data = ws_source.Range("B11:R18").Value
            ws_dest.Range("A1:Q8").Value = data

            # Run macro from PERSONAL.XLSB
            excel.Application.Run("PERSONAL.XLSB!chirag_macro")

            wb.Save()
            wb.Close()

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

# Quit Excel when done
excel.Quit()
print("Done: data copied and macro executed.")
