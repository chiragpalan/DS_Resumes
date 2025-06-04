import os
import win32com.client

# Folder containing your .xlsm files
folder_path = r"C:\path\to\your\xlsm_files"

# Your macro code as a string
vba_code = """
Sub chirag_macro()
    ' Example macro logic
    MsgBox "Macro successfully ran in " & ActiveWorkbook.Name
End Sub
"""

# Launch Excel
excel = win32com.client.Dispatch("Excel.Application")
excel.Visible = False

for file_name in os.listdir(folder_path):
    if file_name.endswith(".xlsm"):
        file_path = os.path.join(folder_path, file_name)
        print(f"📄 Processing: {file_name}")

        try:
            wb = excel.Workbooks.Open(file_path)

            # Check if required sheets exist
            sheet_names = [sheet.Name for sheet in wb.Sheets]
            if "NEW" not in sheet_names or "Sheet2" not in sheet_names:
                print("❌ Skipping: 'NEW' or 'Sheet2' sheet not found.")
                wb.Close(False)
                continue

            ws_source = wb.Sheets("NEW")
            ws_dest = wb.Sheets("Sheet2")

            # Copy B11:R18 to A1:Q8
            ws_dest.Range("A1:Q8").Value = ws_source.Range("B11:R18").Value

            # Inject macro code into a new module (if not already there)
            vb_component = wb.VBProject.VBComponents.Add(1)  # 1 = vbext_ct_StdModule
            vb_component.Name = "ChiragMacroModule"
            vb_component.CodeModule.AddFromString(vba_code)

            # Activate before running macro
            wb.Activate()
            ws_dest.Activate()

            # Run the newly injected macro
            excel.Application.Run(f"{wb.Name}!chirag_macro")

            wb.Save()
            wb.Close()
            print("✅ Done")

        except Exception as e:
            print(f"❌ Error in {file_name}: {e}")

excel.Quit()
print("🎉 All files processed.")
