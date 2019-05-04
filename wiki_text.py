import wikipediaapi


wiki = wikipediaapi.Wikipedia(
    language="de",
    extract_format=wikipediaapi.ExtractFormat.WIKI
)

page_wiki = wiki.page("Albert Einstein")
print("Page - Title: %s" % page_wiki.title)
print("Page - Summary: %s" % page_wiki.summary[0:60])
print("Page-Text: ", page_wiki.text[0:100])
