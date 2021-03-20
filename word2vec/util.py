import re
def clean_line(line):
    pattern = re.compile(r"(\w+\.\w+|\w+)", re.UNICODE) # word or word with . in between
    return tuple(filter(lambda x: pattern.search(x), line.strip().lower().split()))
