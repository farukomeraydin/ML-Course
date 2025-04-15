from sklearn.feature_extraction.text import CountVectorizer

texts = ['Film güzeldi ama oyuncular kötü oynamıştı. Senaryo da güzeldi.', 'Film berbattı. Ama ben eğlendim. Film beklenedildiği gibi bitti.']

cv = CountVectorizer(dtype='uint8')

cv.fit(texts)

print(cv.vocabulary_)

sparse_result = cv.transform(texts)
dense_result = sparse_result.todense()
print(dense_result)
