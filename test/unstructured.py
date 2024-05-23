from unstructured_client import UnstructuredClient
from unstructured_client.models import shared
from unstructured_client.models.errors import SDKError
import pickle
client = UnstructuredClient(api_key_auth="mr8Maj49GKr9rCyUpxa3VXG98XPnI7")
filename = "../resume.pdf"

with open(filename, "rb") as f:
    files=shared.Files(
        content=f.read(),
        file_name=filename,
    )

req = shared.PartitionParameters(files=files)

try:
    resp = client.general.partition(req)
    pickle.dump(resp, file=open("partitioned.pkl", "wb"))
except SDKError as e:
    print(e)
