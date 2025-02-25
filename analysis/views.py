import base64
from io import BytesIO
from django.shortcuts import render
from django.http import HttpResponse
from .forms import ImageUploadForm
from .dummy_ai import dummy_ai_module
from PIL import Image
import openpyxl

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
    ws.title = "Analysis"
    
    ws.append(["Filename", result.get("filename")])
    ws.append([])
    ws.append(["Metric", "Value"])
    ws.append(["Vascular Area", result.get("vascular_area")])
    ws.append(["Vascular Diameter", result.get("vascular_diameter")])
    ws.append(["Xylem Count", result.get("xylem_count")])
    ws.append(["Xylem Diameter", result.get("xylem_diameter")])
    
    ws_details = wb.create_sheet(title="Xylem Details")
    ws_details.append(["ID", "Area", "Diameter"])
    for detail in result.get("xylem_details", []):
        ws_details.append([detail.get("id"), detail.get("area"), detail.get("diameter")])
    
    stream = BytesIO()
    wb.save(stream)
    stream.seek(0)
    
    response = HttpResponse(
        stream,
        content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
    response['Content-Disposition'] = f'attachment; filename=analysis_{index+1}.xlsx'
    return response


def download_all_analysis(request):
    results = request.session.get('analysis_results')
    if not results:
        return HttpResponse("No analyses available for download.", status=400)
    
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Analyses Summary"
    
    ws.append(["Analysis #", "Filename", "Vascular Area", "Vascular Diameter", "Xylem Count", "Xylem Diameter"])
    for i, result in enumerate(results, start=1):
        ws.append([
            i,
            result.get("filename"),
            result.get("vascular_area"),
            result.get("vascular_diameter"),
            result.get("xylem_count"),
            result.get("xylem_diameter")
        ])
    
    for i, result in enumerate(results, start=1):
        ws_detail = wb.create_sheet(title=f"Analysis {i} Xylem")
        ws_detail.append(["ID", "Area", "Diameter"])
        for detail in result.get("xylem_details", []):
            ws_detail.append([detail.get("id"), detail.get("area"), detail.get("diameter")])
    
    stream = BytesIO()
    wb.save(stream)
    stream.seek(0)
    
    response = HttpResponse(
        stream,
        content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
    response['Content-Disposition'] = 'attachment; filename=all_analyses.xlsx'
    return response
