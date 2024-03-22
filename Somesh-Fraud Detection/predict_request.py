import requests

url = 'http://localhost:5000/predict'
data = {
    'features': [228, 5, 117563.1100, 0.0000, 208908.4100, 0]
}

response = requests.post(url, json=data)
print(response.json())
