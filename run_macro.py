import os
import win32com.client

# === Set folder containing your .xlsm files ===
folder_path = r"C:\path\to\your\xlsm_files"

# === VBA macro code to inject ===
vba_code = """
Sub chirag_macro()
    MsgBox "chirag_macro ran successfully on " & ActiveWorkbook.Name
End Sub
"""

# Start Excel
excel = win32com.client.Dispatch("Excel.Application")
excel.Visible = False  # Set True to see Excel during processing

for file_name in os.listdir(folder_path):
    if file_name.endswith(".xlsm"):
        file_path = os.path.join(folder_path, file_name)
        print(f"📄 Processing: {file_name}")

        try:
            wb = excel.Workbooks.Open(file_path)

            # Check for 'NEW' sheet
            sheet_names = [s.Name for s in wb.Sheets]
            if "NEW" not in sheet_names:
                print("❌ Skipping: 'NEW' sheet not found.")
                wb.Close(False)
                continue

            ws_source = wb.Sheets("NEW")

            # Create 'Sheet2' if it doesn't exist
            if "Sheet2" not in sheet_names:
                ws_dest = wb.Sheets.Add(After=wb.Sheets(wb.Sheets.Count))
                ws_dest.Name = "Sheet2"
                print("🆕 'Sheet2' created.")
            else:
                ws_dest = wb.Sheets("Sheet2")

            # Copy data from NEW!B11:R18 to Sheet2!A1:Q8
            ws_dest.Range("A1:Q8").Value = ws_source.Range("B11:R18").Value

            # Inject macro into workbook
            vb_component = wb.VBProject.VBComponents.Add(1)  # 1 = Standard Module
            vb_component.Name = "ChiragMacroModule"
            vb_component.CodeModule.AddFromString(vba_code)

            # Run the macro
            wb.Activate()
            ws_dest.Activate()
            excel.Application.Run(f"{wb.Name}!chirag_macro")

            wb.Save()
            wb.Close()
            print("✅ Done")

        except Exception as e:
            print(f"❌ Error processing {file_name}: {e}")

excel.Quit()
print("🎉 All files processed.")
