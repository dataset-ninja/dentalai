Dataset **Dentalai** can be downloaded in [Supervisely format](https://developer.supervisely.com/api-references/supervisely-annotation-json-format):

 [Download](https://assets.supervisely.com/supervisely-supervisely-assets-public/teams_storage/m/F/wP/t382Lzi0RfoQW2Ej3bKF1h4GobQKA78zvoaBxA9LK8QJB3ijuJyqBb1J3iQAZ3s47hgvI4q4wcJ3amf1qiOzSxQxzTBSicjDQ6GdffP0NnflgfxRVOcNRtRJMxz1.tar)

As an alternative, it can be downloaded with *dataset-tools* package:
``` bash
pip install --upgrade dataset-tools
```

... using following python code:
``` python
import dataset_tools as dtools

dtools.download(dataset='Dentalai', dst_dir='~/dataset-ninja/')
```
Make sure not to overlook the [python code example](https://developer.supervisely.com/getting-started/python-sdk-tutorials/iterate-over-a-local-project) available on the Supervisely Developer Portal. It will give you a clear idea of how to effortlessly work with the downloaded dataset.

