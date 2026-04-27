import re


TOKEN_PATTERN = re.compile(r"(?=[a-zA-Z0-9\u0600-\u06FF]*[a-zA-Z\u0600-\u06FF])[a-zA-Z0-9\u0600-\u06FF]+")


def preprocess(text, min_token_length=2):
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove emails
    text = re.sub(r"\S+@\S+", " ", text)

    # Keep Arabic text and Latin/Arabizi words such as "b7al" and "3lach".
    # Pure numbers are ignored because they are rarely useful as language-model tokens.
    tokens = TOKEN_PATTERN.findall(text)

    # Remove very short tokens
    tokens = [token for token in tokens if len(token) >= min_token_length]

    return tokens
