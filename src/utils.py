# utils.py

def send_email_html(src, dst, subject, html):
    import boto3
    client = boto3.Session(profile_name='trinity', region_name='us-east-1').client('ses')
    client.send_email(
        Source=src,
        SourceArn='arn:aws:ses:us-east-1:886656810413:identity/%s' % src,
        Destination={
            'ToAddresses': [
                dst,
            ],
        },
        Message={
            'Subject': {
                'Data': subject,
                'Charset': 'UTF-8'
            },
            'Body': {
                'Html': {
                    'Data': html,
                    'Charset': 'UTF-8'
                },
            }
        },
    )

def upload_to_s3(s3, bucket, filename):
    data = open(filename, 'rb')
    s3.Bucket(bucket).put_object(Key=filename, Body=data)
