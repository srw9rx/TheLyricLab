from django.shortcuts import render
from django.http import JsonResponse

# Create your views here.

def home(request):
    # Do some processing here
    # context = {'message': 'Hello, world!'}
    # Render the HTML template with the given context data
    return render(request, 'home') # context=context)

def generateRap(prompt):
    # your python code here
    return f"Rap generated here, using prompt {prompt}"

def run_generate_rap(request):
    prompt = request.GET.get('prompt')
    print('prompt is ', prompt)
    output = generateRap(prompt)
    print('output is ', output)
    return JsonResponse({'output': output})
