from django.shortcuts import render
from .forms import ImageUploadForm
from .dummy_ai import dummy_ai_module
from PIL import Image
import base64

def upload_images(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            images = form.cleaned_data['images']
            results = []
            for img in images:
                # Validate that the file is an image
                try:
                    Image.open(img)
                except Exception as e:
                    print(f"File {img.name} is not a valid image: {e}")
                    continue

                ai_result = dummy_ai_module(img)
                
                # Convert ContentFile objects to base64 strings for inline display
                processed_b64 = base64.b64encode(ai_result["processed_file"].read()).decode('utf-8')
                ai_result["processed_file"].seek(0)
                vascular_b64 = base64.b64encode(ai_result["vascular_file"].read()).decode('utf-8')
                ai_result["vascular_file"].seek(0)
                total_root_b64 = base64.b64encode(ai_result["total_root_file"].read()).decode('utf-8')
                ai_result["total_root_file"].seek(0)
                xylem_b64 = base64.b64encode(ai_result["xylem_file"].read()).decode('utf-8')
                ai_result["xylem_file"].seek(0)
                
                result_dict = {
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
            return render(request, 'results.html', {'results': results})
        else:
            print(form.errors)
    else:
        form = ImageUploadForm()
    return render(request, 'upload.html', {'form': form})
