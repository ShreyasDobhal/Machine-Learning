from nltk.stem.snowball import SnowballStemmer

stemmer=SnowballStemmer("english")

print(stemmer.stem("running"))
print(stemmer.stem("thinking"))
print(stemmer.stem("compiling"))
print(stemmer.stem("finding"))