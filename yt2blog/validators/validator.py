import re

def validateYoutubeLink(url):
    if not len(url.strip()):
        raise Exception("Empty url provided.")
    
    # youtube_regex = re.compile(
    #     r'^(https?://)?(www\.)?(youtube\.com|youtu\.be)/(watch\?v=|playlist\?|embed/|[a-zA-Z0-9_-]+)$'
    # )
    youtube_regex = re.compile(
        r'^(https?://)?(www\.)?(youtube\.com|youtu\.be)/((watch\?v=[a-zA-Z0-9_-]+(&.*)?$)|(embed/[a-zA-Z0-9_-]+$)|([a-zA-Z0-9_-]+$))'
    )


    if not bool(youtube_regex.match(url)):
        raise Exception("Not a YouTube link.")


def validateEmail(email):
    pattern = r'^[\w\.-]+@[a-zA-Z\d-]+\.[a-zA-Z]{2,}$'
    if re.match(pattern, email) is None:
        raise Exception("Invalid email format.")
