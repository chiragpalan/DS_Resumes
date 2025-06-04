import os
import win32com.client

folder_path = r"C:\path\to\your\xlsxfiles"

excel = win32com.client.Dispatch("Excel.Application")
excel.Visible = False

for file in os.listdir(folder_path):
    if file.endswith(".xlsx"):
        file_path = os.path.join(folder_path, file)
        wb = excel.Workbooks.Open(file_path)

        new_path = file_path[:-5] + ".xlsm"  # Replace .xlsx with .xlsm
        wb.SaveAs(new_path, FileFormat=52)  # 52 = .xlsm format
        wb.Close()

excel.Quit()
