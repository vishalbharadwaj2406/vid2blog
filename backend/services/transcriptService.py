from youtube_transcript_api import YouTubeTranscriptApi
from utils.urlUtils import getYoutubeVideoID
import json



def getTranscript(url):

    video_id = getYoutubeVideoID(url)
    if video_id is None:
        raise Exception("Invalid video ID.")
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    
    return transcript
