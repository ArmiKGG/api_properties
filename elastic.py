import os
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

load_dotenv()


def connect_elasticsearch():
    _es = None
    _es = Elasticsearch(
        [os.environ['EZA_ML_ES_HOST_URL']],
        http_auth=(os.environ['EZA_ML_LOGIN'], os.environ['EZA_ML_PASSWORD']),
        scheme="https",
        port=int(os.environ['EZA_ML_ES_PORT'])
    )
    if _es.ping():
        print('ES connected')
    else:
        print('Could not connect to ES!')
    return _es


def get_last_scan_date(es_client):
    body_query = {
        "query": {
            "match_all": {}
        },
        "size": 1,
        "sort": [
            {
                "scanDate": {
                    "order": "desc"
                }
            }
        ]
    }
    last_scan = es_client.search(index="scans", body=body_query)
    return last_scan['hits']['hits'][0]['_source']['scanDate']


def get_houses(es_client, last_scan_date):
    print(last_scan_date)
    body_q = {
        "query": {
            "bool": {
                "filter": [
                    {
                        "exists": {
                            "field": "images.path"
                        }
                    },
                    {
                        "exists": {
                            "field": "description"
                        }
                    }
                ],
                "must_not": [
                    {
                        "exists": {
                            "field": "images.classes"
                        }
                    }
                ]
            }
        }
    }
    size = int(os.environ['EZA_ML_ES_PAGE_SIZE'])
    response_to_es = es_client.search(index='aggregated-properties', scroll=os.environ['EZA_ML_SCROLL'], size=size,
                                      body=body_q)

    return response_to_es
