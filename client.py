import requests
import json

gradients = [6, 3, 2] # Replace with your own gradients
url = "http://localhost:8000/"
data = {'gradients': gradients}
response = requests.post(url, data=json.dumps(data), headers={'Content-type': 'application/json'})
averaged_gradient = json.loads(response.content)['averaged_gradient']
