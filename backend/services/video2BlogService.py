from services import transcriptService, AIService, extractorService, emailService
import math
from constants import config
import time


def createKnowledge(transcript, chunk_duration, overlap_duration):
    chunk_duration = chunk_duration * 60
    overlap_duration = overlap_duration * 60

    result = []
    current_chunk_start = 0
    current_chunk_end = chunk_duration

    while current_chunk_start < transcript[-1]['start'] + transcript[-1]['duration']:
        chunk_text = []

        for entry in transcript:
            entry_start = entry['start']
            entry_end = entry_start + entry['duration']
            if (entry_start < current_chunk_end and entry_end > current_chunk_start):
                overlap_start = max(current_chunk_start, entry_start)
                overlap_end = min(current_chunk_end, entry_end)
                if overlap_start < overlap_end:
                    chunk_text.append(entry['text'])

        if chunk_text:
            result.append(" ".join(chunk_text))

        current_chunk_start += chunk_duration - overlap_duration
        current_chunk_end = current_chunk_start + chunk_duration

    return result


def createKnowledgeWithTimestamps(transcript, chunk_duration, overlap_duration):

    # Convert chunk_duration and overlap_duration from minutes to seconds
    chunk_duration = chunk_duration * 60
    overlap_duration = overlap_duration * 60

    def format_time(seconds):
        """Convert seconds to hh:mm:ss format."""
        hours = math.floor(seconds // 3600)
        minutes = math.floor((seconds % 3600) // 60)
        seconds = math.floor(seconds % 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"

    result = {}
    current_chunk_start = 0
    current_chunk_end = chunk_duration

    while current_chunk_start < transcript[-1]['start'] + transcript[-1]['duration']:
        chunk_text = []

        for entry in transcript:
            entry_start = entry['start']
            entry_end = entry_start + entry['duration']
            if entry_start < current_chunk_end and entry_end > current_chunk_start:
                overlap_start = max(current_chunk_start, entry_start)
                overlap_end = min(current_chunk_end, entry_end)
                if overlap_start < overlap_end:  # There's some overlap
                    chunk_text.append(entry['text'])

        if chunk_text:
            start_time = format_time(current_chunk_start)
            end_time = format_time(current_chunk_end)
            timestamp_key = f"{start_time} - {end_time}"
            result[timestamp_key] = " ".join(chunk_text)

        current_chunk_start += chunk_duration - overlap_duration
        current_chunk_end = current_chunk_start + chunk_duration

    return result


def getSummaries(chunks):
    return [AIService.rephrase(chunk, config['LLM']) for chunk in chunks]


def concatenateSummaries(summaries, chunks_with_ts):
    concatenated_summaries = ""
    for ind, key in enumerate(chunks_with_ts):
        concatenated_summaries += f"{key}\n{summaries[ind]}\n\n"
    return concatenated_summaries


def processBlog(url, email):
    try:
        transcript = transcriptService.getTranscript(url)

    except Exception as e:
        raise e

    chunks = createKnowledge(transcript, config['CHUNK_DURATION'], config['OVERLAP_DURATION'])
    chunks_with_ts = createKnowledgeWithTimestamps(transcript, config['CHUNK_DURATION'], config['OVERLAP_DURATION'])

    summaries = getSummaries(chunks)

    concated_summaries = concatenateSummaries(summaries, chunks_with_ts)

    blog = AIService.generateBlog(concated_summaries, config['LLM'])
    code = extractorService.getCode(transcript)
    blog_with_code = AIService.generateBlogWithCode(blog, code)
    if blog_with_code.startswith("```markdown"):
        blog_with_code = blog_with_code[11:-4]
    emailService.sendMarkdownEmail(email, "The Blog You Requested For", blog_with_code)
