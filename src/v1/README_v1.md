### Web Recommendation System for RynekPierwotny.pl

### General information
+ *web_recommendation.api*:
    - **api.py** - API definitions
+ *web_recommendation.model*:
    - **prediction.py** - the file that will be used in the API and is responsible for generating recommendations
+ *web_recommendation.managers*:
    - **db_manager.py** - functions for fetching and running sql queries
    - **generate_data.py** - functions for aggregating users activity on the website
    - **load_file** - functions for handling aggregates in a pickle format
    - **manager** - helper functions mainly responsible for finding the id in the database
    - **init_pickle_update.py** - the script is run right after starting docker. The goal is to download data marts:
views and applications from db
    - **knn_data_processing.py** - knn calculations for the KNN model applied in GH
+ *web_recommendation.consts*:
    - **config** - database access
    - **const** - constant parameters such as threshold
    - **paths** - paths to data marts
    - **data** - logs (web recommendation time)
+ *web_recommendation.consts*:
    - **model** - The folder contains files with models, logically divided into Gethome and RynekPierwotny and
  a file that combines them into one named as `prediction.py`.
+ *sql*:
    - **scripts** - the sql scripts for the project


### Instruction for run api serving recommendations
0. Complete the missing fields in " `docker_conf/var.env`.
```commandline
db_username
db_password
db_gh_username
db_gh_password
```
1. Build containers.
```bash
docker-compose build
```
2. Run production server.
```bash
docker-compose up
```

### An example of using the API
```python
import requests

url = "http://0.0.0.0:8000/web-recommendation/user"

payload = {'user_id': '<user_id_from_users_user>',
           'logged_in': True,
           'webhook_url': 'https://localhost',
           'portal': 'both'}

# when user is not logged in
# payload = {'user_id': '<session_id>', 'logged_in': False, 'webhook_url': 'https://localhost', 'portal': 'both'}

headers = {'Authorization': 'Bearer 11'}
response = requests.request("POST", url, headers=headers, data=payload)
print(response.text)
```
Inside the `payload` it is needed to set below parameters:
- `logged_in` - `True` if the user is logged in, `False` if not. By default, it is assumed that the user is logged in
- `user_id` - id from table `users_user` if user is logged in, `session_id` if not
- `webhook_url` - URL where the results will be sent
- `portal` - it is possible to select the portal for which the recommendations will be generated. In default is set as both, but there is an option to select only rp or gh

As the output You will get the JSON:
```json
{'rp_properties': [list of max TOP-N property_id],
'gh_properties': [list of max TOP-N offer_id from mdb_offers_offer]}}
```
Recommendations for viewing will be available in docker logs.
