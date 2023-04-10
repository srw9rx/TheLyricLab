from django.shortcuts import render

# Create your views here.

def home(request):
    # Do some processing here
    # context = {'message': 'Hello, world!'}
    # Render the HTML template with the given context data
    return render(request, 'home.html') # context=context)
