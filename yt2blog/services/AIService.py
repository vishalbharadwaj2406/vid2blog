from services import gptService

def rephrase(transcript_chunk, llm='gpt'):
    if llm == 'gpt':
        summaries = gptService.rephrase(transcript_chunk)

    return summaries

def generateBlog(concated_summaries, llm='gpt'):
    if llm == 'gpt':
        blog = gptService.generateBlog(concated_summaries)

    return blog

def generateBlogWithCode(blog_without_code, code, llm='gpt'):
    if llm == 'gpt':
        blog_with_code = gptService.generateBlogWithCode(blog_without_code, code)

    return blog_with_code
