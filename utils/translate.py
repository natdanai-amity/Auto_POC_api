from pythainlp.translate import Translate

def thai2en(text):
    th2en = Translate('th', 'en')
    text_out = th2en.translate(text)
    return text_out

def en2thai(text):
    en2th = Translate('en', 'th')
    text_out = en2th.translate(text)
    return text_out