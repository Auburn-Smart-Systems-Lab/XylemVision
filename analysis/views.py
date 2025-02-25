from django.shortcuts import render
from .forms import ImageUploadForm
from .models import ProcessedImage
from .dummy_ai import dummy_ai_module

def upload_images(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            images = request.FILES.getlist('images')
            results = []
            for img in images:
                ai_result = dummy_ai_module(img)
                
                processed_instance = ProcessedImage(
                    original_image=img,
                    vascular_area=ai_result["vascular_area"],
                    vascular_diameter=ai_result["vascular_diameter"],
                    xylem_count=ai_result["xylem_count"],
                    xylem_diameter=ai_result["xylem_diameter"],
                    xylem_details=ai_result["xylem_details"],
                )
                processed_instance.processed_image.save('processed.jpg', ai_result["processed_file"])
                processed_instance.vascular_bundle_image.save('vascular.jpg', ai_result["vascular_file"])
                processed_instance.total_root_image.save('total_root.jpg', ai_result["total_root_file"])
                processed_instance.xylem_image.save('xylem.jpg', ai_result["xylem_file"])
                processed_instance.save()
                results.append(processed_instance)
            return render(request, 'results.html', {'results': results})
    else:
        form = ImageUploadForm()
    return render(request, 'upload.html', {'form': form})
