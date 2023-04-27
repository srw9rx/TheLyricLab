#Django framework implementation
from django.shortcuts import render
from django.http import JsonResponse
from django.http import HttpResponseRedirect
from django.urls import reverse
#imports for Rap Generation
import torch
from transformers import GPT2LMHeadModel,  GPT2Tokenizer

# Create your views here.
def index(request):
    return HttpResponseRedirect(reverse('home'))

def indexrap(request):
    return HttpResponseRedirect(reverse('run-generate-rap'))

def home(request):
    # Do some processing here
    # context = {'message': 'Hello, world!'}
    # Render the HTML template with the given context data
    return render(request, 'home') # context=context)

def generateRap(prompt):
    #set device for ML modeling
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    #load the model and add the path file
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|endoftext|>', eos_token='<|endoftext|>', pad_token='<|endoftext|>', padding='max_length', truncation=True)
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    model = torch.load('rapgenmid.pt', map_location=device)
    model.eval()
    
    #encode the text for generation
    encodings_dict = tokenizer.encode(('<|endoftext|>'+ prompt), truncation=True, max_length=32)
    generated = torch.tensor(encodings_dict).unsqueeze(0)
    generated = generated.to(device)

    #perform the generation
    with torch.no_grad():
        sample_outputs = model.generate(generated, top_k=50, top_p=0.90, num_return_sequences=1, max_new_tokens=768)

    return tokenizer.decode(sample_outputs[0], skip_special_tokens=True)

def run_generate_rap(request):
    prompt = request.GET.get('prompt')
    print('prompt is ', prompt)
    output = generateRap(prompt)
    print('output is ', output)
    return JsonResponse({'output': output})
