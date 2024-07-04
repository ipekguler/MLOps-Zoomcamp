import os
import sys
import pandas as pd
import batch

year = int(sys.argv[1])
month = int(sys.argv[2])

S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL')
output_file = batch.get_output_path(year, month)

if S3_ENDPOINT_URL is not None:
    options = {
        'client_kwargs': {
            'endpoint_url': S3_ENDPOINT_URL
        }
    }
    df = pd.read_parquet(output_file, storage_options=options)
else:
    df = pd.read_parquet(output_file)

print(df.predicted_duration.sum())