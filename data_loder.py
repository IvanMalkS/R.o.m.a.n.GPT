import os
import json
import requests

def fetch_data():
    url = 'https://api.eternity-business.ru/web/v1/anal/analyzed'
    headers = {
        'Authorization': 'Bearer NhrWATjSXlckMTFz'
    }
    params = {
        'page': 1,  # Starting from the first page
        'limit': 500
    }

    data = []  # Initialize list to store data

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        pages = int(response.json()['_meta']['pageCount'])  # Get the number of pages
        data += response.json()['items']  # Add data from the first page

        for i in range(2, pages + 1):  # Iterate through all other pages
            params['page'] = i
            response = requests.get(url, headers=headers, params=params)

            if response.status_code == 200:
                data += response.json()['items']  # Add data from the next page
        return data
    else:
        print(f'Request failed with status code {response.status_code}')
        return None

def load_data(file_path='cached_data.json'):
    if os.path.exists(file_path):
        print("Loading data from cache...")
        with open(file_path, 'r') as f:
            data = json.load(f)
    else:
        print("Fetching data from API...")
        data = fetch_data()
        if data:
            with open(file_path, 'w') as f:
                json.dump(data, f)
    return data
