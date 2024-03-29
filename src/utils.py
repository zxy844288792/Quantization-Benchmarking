# utils.py
import numpy as np

def post_processing(img):
    if img.shape[1] == 1001:
        return img[:,1:]
    return img

def crop_center(img, threshold=0.875):
    y,x,z = img.shape
    startx = int((x - x * threshold) / 2)
    starty = int((y - y * threshold) / 2)
    x_size = x - startx * 2
    y_size = y - starty * 2
    return img[starty:starty+y_size,startx:startx+x_size,:]


def send_email_html(src, dst, subject, html, arn):
    import boto3
    client = boto3.Session(profile_name='trinity', region_name='us-east-1').client('ses')
    client.send_email(
        Source=src,
        SourceArn= arn + ('%s' % src),
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
