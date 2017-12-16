
# Bag of Words

# Frequency count of occurance of words

from sklearn.feature_extraction.text import CountVectorizer

vectorizer=CountVectorizer()
string1="Not all good things lasts forever and Deep ; our 50% Graduation Scholarship Program is also coming to an end"
string2="Hey Participate Not and compete with Participate yourself in these exciting challenges to  win cool prizes"
string3="You’re invited to free Participate in a free preview of our Deep Learning Nanodegree Foundation program classroom"
email_list=[string1,string2,string3]
#bagOfWords=vectorizer(email_list)
bagOfWords=vectorizer.fit(email_list)
bagOfWords=vectorizer.transform(email_list)
print(bagOfWords)
print(vectorizer.vocabulary_.get("Participate "))
