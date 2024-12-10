import re
from urllib.parse import urlparse, parse_qs

def getYoutubeVideoID(url):
    """
    Extracts the YouTube video ID from a given URL.

    Args:
        url (str): The YouTube URL.

    Returns:
        str: The video ID if found, otherwise None.
    """
    # Parse the URL
    parsed_url = urlparse(url)
    
    # Check for standard YouTube URL with query parameters
    if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
        query_params = parse_qs(parsed_url.query)
        return query_params.get('v', [None])[0]
    
    # Check for shortened YouTube URL
    if parsed_url.hostname == 'youtu.be':
        return parsed_url.path.strip('/')
    
    # Fallback: try to extract video ID using regex
    match = re.search(r'(v=|\/)([a-zA-Z0-9_-]{11})', url)
    if match:
        return match.group(2)
    
    return None