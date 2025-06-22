from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.core.files.storage import default_storage
from django.conf import settings
import json
import os  # ✅ Required for path handling
from . import ai_assistant  # your AI logic

@csrf_exempt
def assistant_api(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body.decode("utf-8"))
            question = data.get("question")
            pdf_path = data.get("pdf_path")  # ✅ Read from request

            if not question or not pdf_path:
                return JsonResponse({"error": "Missing fields."}, status=400)

            vstore, text = ai_assistant.get_vectorstore(pdf_path)  # ✅ Use dynamic PDF
            pdf_qa, fallback_qa, full_text = ai_assistant.get_chains(vstore, text)
            answer = ai_assistant.answer_query(pdf_qa, fallback_qa, full_text, question)

            return JsonResponse({"answer": answer})
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON."}, status=400)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    else:
        return JsonResponse({"error": "Only POST method allowed."}, status=405)

@csrf_exempt
def upload_pdf(request):
    if request.method == "POST":
        if 'pdf' not in request.FILES:
            return JsonResponse({"error": "No PDF file provided."}, status=400)

        pdf_file = request.FILES['pdf']
        save_path = os.path.join(settings.MEDIA_ROOT, pdf_file.name)

        with open(save_path, 'wb+') as destination:
            for chunk in pdf_file.chunks():
                destination.write(chunk)

        return JsonResponse({
            "message": "PDF uploaded successfully!",
            "pdf_path": save_path
        })
    return JsonResponse({"error": "Only POST method allowed."}, status=405)
