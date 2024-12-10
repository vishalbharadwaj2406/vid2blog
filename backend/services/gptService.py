from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()


def rephrase(transcript_chunk):
    prompt = f"""
            {transcript_chunk}\n\n
            Rephrase, in detail, the above chunk of text comprehensively while retaining all key implementation details.
            This chunk may contain description of code. Pay close attention to the details of the code description.
            Maintain chronology of the explanation. Maintain all technical details, the way they are presented in the text. Do not lose any detail.
            In the video, the speaker is speaking in a conversational tone, your job is to make it a coherent and formal explanation. Write it in the tone of a blog post.
            Keep the explanation verbose and comprehensive.
             """
    response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt}
            ],
            seed=42,
            temperature=0.7
        )
    return response.choices[0].message.content


def generateBlog(concatenated_summaries):
    blog_prompt = f"""{concatenated_summaries}
            The above text is the transcript of a video. I need you to rephrase it. The above text follows this format:
            '<hh:mm:ss - hh:mm:ss>
            <content part 1>
            <hh:mm:ss - hh:mm:ss>
            <content part 2>
            ...'

            '<hh:mm:ss - hh:mm:ss>' denotes the timestamp range and the content is the transcribed text corresponding to that timestamp range.
            There might be overlaps of content between consecutive timestamp ranges. For example:
            00:00:00 - 00:10:00
            <content part 1>
            00:07:00 - 00:20:00
            <content part 2>
            ...

            Here, there is an overlap between the first 3 minutes of part 2 and the last 3 minutes of part 1. So, the last few lines of part 1's content may be repeated in the first few lines of part 2's content.
            Rephrase the entire transcript in a coherent manner while ensuring smooth transitions between consecutive parts by handling overlapping content.
            Along with rephrasing, I need you to add appropriate titles whenever there is a natural transition in the topic of discussion.
            Make sure to maintain all technical details, including explanations of code.
            The output has to be in a markdown format.

    """
    response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user","content":  blog_prompt}
                ],
                seed=42,
                temperature=0.7
        )

    return response.choices[0].message.content


def generateBlogWithCode(blog_without_code, code):
    code_prompt = f"""{blog_without_code}
                  The above text is a blog.
                  {code}
                  The above is the supplementary code for the blog. The code is not a continuous coherent program, it is a concatenation of various snippets.
                  Interweave the code snippets with the blog in appropriate sections.
                  If there are any existing code snippets in the original blog, replace them with the right code snippets from the code provided.
                  Ensure the final blog is coherent and comprehensive. Insert the code snippets as code blobks amidst the blog content.
                  Make sure the code snippet aligns appropriately with the content being described in the corresponding section.
                  The output has to be in a markdown format.
                """

    response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user","content":  code_prompt}
                ],
                seed=42,
                temperature=0.7
            )

    return response.choices[0].message.content


