from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, EmotionOptions


def sentiment(text):
    api_key = '4AclcErHA5srDiyQzFCRIeihcGW0Ei_JgUDmDaVxwftj'
    url = 'https://api.us-south.natural-language-understanding.watson.cloud.ibm.com/instances/bd387bb8-806b-4d65-ac70-6e6f95791c40'

    authenticator = IAMAuthenticator(api_key)
    natural_language_understanding = NaturalLanguageUnderstandingV1(
        version='2020-08-01',
        authenticator=authenticator
    )
    natural_language_understanding.set_service_url(url)

    response = natural_language_understanding.analyze(
        text=text,
        features=Features(emotion=EmotionOptions())).get_result()

    sadness = response['emotion']['document']['emotion']['sadness']
    joy = response['emotion']['document']['emotion']['joy']
    fear = response['emotion']['document']['emotion']['fear']
    disgust = response['emotion']['document']['emotion']['disgust']
    anger = response['emotion']['document']['emotion']['anger']

    print("Sadness: ", sadness)
    print("Joy: ", joy)
    print("Fear: ", fear)
    print("Disgust: ", disgust)
    print("Anger: ", anger)
