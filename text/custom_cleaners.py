import re


def word_replacer(text):
    text = re.sub(r"\bpersona\b", "persowna", text)
    text = re.sub(r"\bfurthermore\b", "firthermore", text)
    text = re.sub(r"\banastasia\b", "anastaysha", text)
    text = re.sub(r"\bruggiero\b", "rujero", text)
    text = re.sub(r"\brobust\b", "rowbust", text)
    text = re.sub(r"\bproclivity\b", "pro-clivity", text)
    text = re.sub(r"\bproactive\b", "pro-active", text)
    text = re.sub(r"\bdistribute\b", "distri-bute", text)
    text = re.sub(r"\breinfeld\b", "rein-feld", text)
    text = re.sub(r"\bal\b", "l", text)
    text = re.sub(r"\bluciano\b", "lewcheeyano", text)
    text = re.sub(r"\bsenator\b", "senitor", text)
    text = re.sub(r"\bkefauver\b", "keyfarver", text)
    text = re.sub(r"\balleged\b", "allejd", text)
    text = re.sub(r"\befforts\b", "effirts", text)
    text = re.sub(r"\bbuchalter\b", "buckhalter", text)
    text = re.sub(r"\bpuglia\b", "pulya", text)
    text = re.sub(r"\bguarino\b", "gwarino", text)
    return text


def word_filter(text):
    text = str(text).lower()
    text = word_replacer(text)
    return text
