import requests

ride = "Trump Said Some INSANELY Racist Stuff Inside The Oval Office, And Witnesses Back It Up"

url = 'http://localhost:9696/predict'
response = requests.post(url, json=ride)
print(response.json())