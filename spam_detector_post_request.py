import requests

url = 'http://127.0.0.1:9856/spam_detection'
payload = {
    'message': """
    Congratulations ! you have won 500 $
    please reach out to us in this number +15526238458 or this link http://link.com
    """
}
headers= {
    'Content-Type' : 'application/json'
}

r= requests.post(url , json = payload , headers = headers )
print(r.text)