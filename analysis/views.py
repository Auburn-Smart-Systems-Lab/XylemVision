import base64
import io
from django.http import HttpResponse
from django.shortcuts import render
from .forms import ImageUploadForm
from .dummy_ai import dummy_ai_module
from PIL import Image
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.drawing.image import Image as OXI

def upload_images(request):
    results = None
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            images = form.cleaned_data['images']
            results = []
            for img in images:
                try:
                    Image.open(img)
                except Exception as e:
                    print(f"File {img.name} is not a valid image: {e}")
                    continue

                ai_result = dummy_ai_module(img)
                
                processed_b64 = base64.b64encode(ai_result["processed_file"].read()).decode('utf-8')
                ai_result["processed_file"].seek(0)
                vascular_b64 = base64.b64encode(ai_result["vascular_file"].read()).decode('utf-8')
                ai_result["vascular_file"].seek(0)
                total_root_b64 = base64.b64encode(ai_result["total_root_file"].read()).decode('utf-8')
                ai_result["total_root_file"].seek(0)
                xylem_b64 = base64.b64encode(ai_result["xylem_file"].read()).decode('utf-8')
                ai_result["xylem_file"].seek(0)
                
                result_dict = {
                    "filename": img.name,
                    "processed_image": processed_b64,
                    "vascular_image": vascular_b64,
                    "total_root_image": total_root_b64,
                    "xylem_image": xylem_b64,
                    "vascular_area": ai_result["vascular_area"],
                    "vascular_diameter": ai_result["vascular_diameter"],
                    "xylem_count": ai_result["xylem_count"],
                    "xylem_diameter": ai_result["xylem_diameter"],
                    "xylem_details": ai_result["xylem_details"],
                }
                results.append(result_dict)
            request.session['analysis_results'] = results
        else:
            print(form.errors)
    else:
        form = ImageUploadForm()
    return render(request, 'upload.html', {'form': form, 'results': results})


def download_all_analysis(request):
    results = request.session.get('analysis_results')
    if not results:
        return HttpResponse("No analyses available for download.", status=400)
    
    wb = openpyxl.Workbook()
    default_sheet = wb.active
    wb.remove(default_sheet)
    
    def add_image(ws, base64_str, cell, width, height):
        img_data = base64.b64decode(base64_str)
        img_io = io.BytesIO(img_data)
        img = OXI(img_io)
        img.width = width
        img.height = height
        ws.add_image(img, cell)
    
    for result in results:
        sheet_name = result.get("filename")
        if len(sheet_name) > 31:
            sheet_name = sheet_name[:31]
        ws = wb.create_sheet(title=sheet_name)
        
        for col in ["A", "B", "C", "D"]:
            ws.column_dimensions[col].width = 25
        for col in ["E", "F", "G", "H"]:
            ws.column_dimensions[col].width = 20
        
        ws.merge_cells("A1:H1")
        header = ws["A1"]
        header.value = f"Analysis Report for {result.get('filename')}"
        header.font = Font(bold=True, size=14, color="FFFFFF")
        header.fill = PatternFill(start_color="007bff", end_color="007bff", fill_type="solid")
        header.alignment = Alignment(horizontal="center", vertical="center")
        ws.row_dimensions[1].height = 30
        
        data_start = 3
        ws["A" + str(data_start)] = "Metric"
        ws["B" + str(data_start)] = "Value"
        ws["A" + str(data_start)].font = Font(bold=True)
        ws["B" + str(data_start)].font = Font(bold=True)
        current_row = data_start + 1
        metrics = [
            ("Vascular Area", result.get("vascular_area")),
            ("Vascular Diameter", result.get("vascular_diameter")),
            ("Xylem Count", result.get("xylem_count")),
            ("Xylem Diameter", result.get("xylem_diameter")),
        ]
        for metric, value in metrics:
            ws["A" + str(current_row)] = metric
            ws["B" + str(current_row)] = value
            current_row += 1
        
        current_row += 1
        ws.merge_cells(start_row=current_row, start_column=1, end_row=current_row, end_column=4)
        cell = ws.cell(row=current_row, column=1)
        cell.value = "Xylem Details"
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill(start_color="28a745", end_color="28a745", fill_type="solid")
        cell.alignment = Alignment(horizontal="center")
        current_row += 1
        ws["A" + str(current_row)] = "ID"
        ws["B" + str(current_row)] = "Area"
        ws["C" + str(current_row)] = "Diameter"
        for col in ["A", "B", "C"]:
            ws[col + str(current_row)].font = Font(bold=True)
        current_row += 1
        for detail in result.get("xylem_details", []):
            ws["A" + str(current_row)] = detail.get("id")
            ws["B" + str(current_row)] = detail.get("area")
            ws["C" + str(current_row)] = detail.get("diameter")
            current_row += 1

        ws["E2"] = "Processed"
        ws["G2"] = "Vascular Bundle Mask"
        ws["E11"] = "Total Root Mask"
        ws["G11"] = "Xylem Mask"
        add_image(ws, result.get("processed_image"), "E3", 100, 80)
        add_image(ws, result.get("vascular_image"), "G3", 100, 80)
        add_image(ws, result.get("total_root_image"), "E12", 100, 80)
        add_image(ws, result.get("xylem_image"), "G12", 100, 80)
        ws.row_dimensions[3].height = 80
        ws.row_dimensions[12].height = 80
    
    stream = io.BytesIO()
    wb.save(stream)
    stream.seek(0)
    response = HttpResponse(
        stream,
        content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
    response['Content-Disposition'] = 'attachment; filename=all_analyses.xlsx'
    return response


def download_analysis(request, analysis_index):
    try:
        index = int(analysis_index) - 1
    except ValueError:
        return HttpResponse("Invalid analysis index.", status=400)

    results = request.session.get('analysis_results')
    if results is None or index < 0 or index >= len(results):
        return HttpResponse("Invalid analysis index or no analyses available.", status=400)
    
    result = results[index]
    wb = openpyxl.Workbook()
    ws = wb.active

    sheet_name = result.get("filename")
    if len(sheet_name) > 31:
        sheet_name = sheet_name[:31]
    ws.title = sheet_name

    for col in ["A", "B", "C", "D"]:
        ws.column_dimensions[col].width = 25
    for col in ["E", "F", "G", "H"]:
        ws.column_dimensions[col].width = 20

    ws.merge_cells("A1:H1")
    header = ws["A1"]
    header.value = f"Analysis Report for {result.get('filename')}"
    header.font = Font(bold=True, size=14, color="FFFFFF")
    header.fill = PatternFill(start_color="007bff", end_color="007bff", fill_type="solid")
    header.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 30

    data_start = 3
    ws["A" + str(data_start)] = "Metric"
    ws["B" + str(data_start)] = "Value"
    ws["A" + str(data_start)].font = Font(bold=True)
    ws["B" + str(data_start)].font = Font(bold=True)
    current_row = data_start + 1
    metrics = [
        ("Vascular Area", result.get("vascular_area")),
        ("Vascular Diameter", result.get("vascular_diameter")),
        ("Xylem Count", result.get("xylem_count")),
        ("Xylem Diameter", result.get("xylem_diameter")),
    ]
    for metric, value in metrics:
        ws["A" + str(current_row)] = metric
        ws["B" + str(current_row)] = value
        current_row += 1

    current_row += 1
    ws.merge_cells(start_row=current_row, start_column=1, end_row=current_row, end_column=4)
    cell = ws.cell(row=current_row, column=1)
    cell.value = "Xylem Details"
    cell.font = Font(bold=True, color="FFFFFF")
    cell.fill = PatternFill(start_color="28a745", end_color="28a745", fill_type="solid")
    cell.alignment = Alignment(horizontal="center")
    current_row += 1
    ws["A" + str(current_row)] = "ID"
    ws["B" + str(current_row)] = "Area"
    ws["C" + str(current_row)] = "Diameter"
    for col in ["A", "B", "C"]:
        ws[col + str(current_row)].font = Font(bold=True)
    current_row += 1
    for detail in result.get("xylem_details", []):
        ws["A" + str(current_row)] = detail.get("id")
        ws["B" + str(current_row)] = detail.get("area")
        ws["C" + str(current_row)] = detail.get("diameter")
        current_row += 1

    def add_image(ws, base64_str, cell, width, height):
        img_data = base64.b64decode(base64_str)
        img_io = io.BytesIO(img_data)
        img = OXI(img_io)
        img.width = width
        img.height = height
        ws.add_image(img, cell)

    ws["E2"] = "Processed"
    ws["G2"] = "Vascular Bundle Mask"
    ws["E11"] = "Total Root Mask"
    ws["G11"] = "Xylem Mask"
    add_image(ws, result.get("processed_image"), "E3", 100, 80)
    add_image(ws, result.get("vascular_image"), "G3", 100, 80)
    add_image(ws, result.get("total_root_image"), "E12", 100, 80)
    add_image(ws, result.get("xylem_image"), "G12", 100, 80)
    ws.row_dimensions[3].height = 80
    ws.row_dimensions[12].height = 80

    stream = io.BytesIO()
    wb.save(stream)
    stream.seek(0)
    response = HttpResponse(
        stream,
        content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
    response['Content-Disposition'] = f'attachment; filename={sheet_name}.xlsx'
    return response
