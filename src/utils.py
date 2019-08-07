# utils.py
def upload_to_s3(s3, bucket, filename):
    data = open(filename, 'rb')
    s3.Bucket(bucket).put_object(Key=filename, Body=data)
