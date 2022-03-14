import os
import requests

url = os.getenv('CIRCLE_PULL_REQUEST')
if url is None:
    print('develop')
else:
    _, _, _, user, repo, _, num = url.split('/')

    new_url = f'https://api.github.com/repos/{user}/{repo}/pulls/{num}'
    base_branch = requests.get(new_url).json().get('base', {}).get('ref', 'master')

    print(base_branch)

