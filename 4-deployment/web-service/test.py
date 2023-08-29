import requests

ride = "Don Jr. Tries To Mock Al Franken’s Resignation, Backfires Immediately"

url = 'http://localhost:9696/predict'
response = requests.post(url, json=ride)
print(response.json())