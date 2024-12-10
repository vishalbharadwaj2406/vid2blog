# returns transcript with blog

def getCode(transcript):
    with open("/mnt/EA0E697A0E6940A5/UIUC/Sem 2/CS 598 TZ/Project/Experiments/code.txt", "r") as f:
        return f.read()
