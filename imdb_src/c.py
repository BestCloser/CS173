import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('sorted.csv')
df2 = pd.read_csv('test.csv')


# tfidf = TfidfVectorizer(ngram_range=(1, 2))
# tfidf.fit(df['review'])
# X = tfidf.transform(df['review'])
# X_test = tfidf.transform(df['review'])

print("binary true:")
cv = CountVectorizer(binary=True)
cv.fit(df['review'])
Y = cv.transform(df['review'])
Y_test = cv.transform(df['review'])
print(Y.shape)


#print("binary true:")
cv2 = CountVectorizer(binary=True)
cv2.fit(df2['review'])
Y_2 = cv2.transform(df2['review'])
Y_2test = cv2.transform(df2['review'])
#print(Y.shape)

# tfidf_n = TfidfVectorizer(ngram_range=(1,2))
# tfidf_n.fit(df['review'])
# X2 = tfidf_n.transform(df['review'])
# X2_test = tfidf_n.transform(df['review'])

# cv_n = CountVectorizer(binary=True, ngram_range=(1,2))
# cv_n.fit(df['review'])
# Y2 = cv_n.transform(df['review'])
# Y2_test = cv_n.transform(df['review'])



#print(X.shape)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

target = [0 if i < 25000 else 1 for i in range(50000)]

# X_train, X_val, y_train, y_val = train_test_split(X, target, train_size = .75)

# print("tfidf:")
# for c in [0.01, 0.05, 0.25, 0.5, 1, 10, 100, 1000]:
# 	lr = LogisticRegression(C=c)
# 	lr.fit(X_train, y_train)
# 	print ("Accuracy for C=%s: %s" % (c, accuracy_score(y_val, lr.predict(X_val))))


X_train, X_val, y_train, y_val = train_test_split(Y, target, train_size = .75)

print("cv:")
for c in [0.01, 0.05, 0.25, 0.5, 1]:
	lr = LogisticRegression(C=c, random_state=None)
	lr.fit(X_train, y_train)
	print ("Accuracy for C=%s: %s" % (c, accuracy_score(y_val, lr.predict(X_val))))



final_model = LogisticRegression(C=0.05)#, random_state=None)
final_model.fit(Y, target)
print ("Final Accuracy: %s" 
       % accuracy_score(target, final_model.predict(Y_test)))

feature_to_coef = {
    word: coef for word, coef in zip(
        cv.get_feature_names(), final_model.coef_[0]
    )
}
for best_positive in sorted(
    feature_to_coef.items(), 
    key=lambda x: x[1], 
    reverse=True)[:5]:
    print (best_positive)
    
for best_negative in sorted(
    feature_to_coef.items(), 
    key=lambda x: x[1])[:5]:
    print (best_negative)




print("binary false:")
cv = CountVectorizer(binary=True)
cv.fit(df['review'])
Y = cv.transform(df['review'])
Y_test = cv.transform(df['review'])
print(Y.shape)


X_train, X_val, y_train, y_val = train_test_split(Y, target, train_size = .75)


print("cv:")
for c in [0.01, 0.05, 0.25, 0.5, 1]:
    lr = LogisticRegression(C=c)#, random_state=None)
    lr.fit(X_train, y_train)
    print ("Accuracy for C=%s: %s" % (c, accuracy_score(y_val, lr.predict(X_val))))



##modifying y test
Y_test = Y_2test

final_model = LogisticRegression(C=0.05, random_state=None)
final_model.fit(Y, target)
print ("Final Accuracy: %s" 
       % accuracy_score(target, final_model.predict(Y_test)))

feature_to_coef = {
    word: coef for word, coef in zip(
        cv.get_feature_names(), final_model.coef_[0]
    )
}
for best_positive in sorted(
    feature_to_coef.items(), 
    key=lambda x: x[1], 
    reverse=True)[:5]:
    print (best_positive)
    
for best_negative in sorted(
    feature_to_coef.items(), 
    key=lambda x: x[1])[:5]:
    print (best_negative)


###










# X_train, X_val, y_train, y_val = train_test_split(X2, target, train_size = .75)

# print("tfidf+bigram:")
# for c in [0.01, 0.05, 0.25, 0.5, 1, 10, 100, 1000]:
#     lr = LogisticRegression(C=c)
#     lr.fit(X_train, y_train)
#     print ("Accuracy for C=%s: %s" % (c, accuracy_score(y_val, lr.predict(X_val))))


# X_train, X_val, y_train, y_val = train_test_split(Y2, target, train_size = .75)

# print("cv+bigram:")
# for c in [0.01, 0.05, 0.25, 0.5, 1]:
#     lr = LogisticRegression(C=c)
#     lr.fit(X_train, y_train)
#     print ("Accuracy for C=%s: %s" % (c, accuracy_score(y_val, lr.predict(X_val))))
