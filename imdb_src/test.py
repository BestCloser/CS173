import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('sorted.csv')
tfidf = TfidfVectorizer()
tfidf.fit(df['review'])
X = tfidf.transform(df['review'])
X_test = tfidf.transform(df['review'])

cv = CountVectorizer(binary=True)
cv.fit(df['review'])
Y = cv.transform(df['review'])
Y_test = cv.transform(df['review'])

#print(X.shape)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

target = [0 if i < 25000 else 1 for i in range(50000)]

X_train, X_val, y_train, y_val = train_test_split(X, target, train_size = .75)

print("tfidf:")
for c in [0.01, 0.05, 0.25, 0.5, 1, 10, 100, 1000]:
	lr = LogisticRegression(C=c)
	lr.fit(X_train, y_train)
	print ("Accuracy for C=%s: %s" % (c, accuracy_score(y_val, lr.predict(X_val))))

print("C=1:")
final_model = LogisticRegression(C=1)
final_model.fit(X, target)
print ("Final Accuracy: %s" 
       % accuracy_score(target, final_model.predict(X_test)))

feature_to_coef = {
    word: coef for word, coef in zip(
        tfidf.get_feature_names(), final_model.coef_[0]
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







print("C=10:")
final_model = LogisticRegression(C=10)
final_model.fit(X, target)
print ("Final Accuracy: %s" 
       % accuracy_score(target, final_model.predict(X_test)))

feature_to_coef = {
    word: coef for word, coef in zip(
        tfidf.get_feature_names(), final_model.coef_[0]
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







X_train, X_val, y_train, y_val = train_test_split(Y, target, train_size = .75)

print("cv:")
for c in [0.01, 0.05, 0.25, 0.5, 1]:
	lr = LogisticRegression(C=c)
	lr.fit(X_train, y_train)
	print ("Accuracy for C=%s: %s" % (c, accuracy_score(y_val, lr.predict(X_val))))

final_model = LogisticRegression(C=0.05)
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









# final_model = LogisticRegression(C=0.05)
# final_model.fit(X, target)
# print ("Final Accuracy: %s" 
#        % accuracy_score(target, final_model.predict(X_test)))
# Final Accuracy: 0.88128








# feature_to_coef = {
#     word: coef for word, coef in zip(
#         vect.get_feature_names(), final_model.coef_[0]
#     )
# }
# for best_positive in sorted(
#     feature_to_coef.items(), 
#     key=lambda x: x[1], 
#     reverse=True)[:5]:
#     print (best_positive)
    
#     ('excellent', 0.9288812418118644)
#     ('perfect', 0.7934641227980576)
#     ('great', 0.675040909917553)
#     ('amazing', 0.6160398142631545)
#     ('superb', 0.6063967799425831)
    
# for best_negative in sorted(
#     feature_to_coef.items(), 
#     key=lambda x: x[1])[:5]:
#     print (best_negative)
    
#     ('worst', -1.367978497228895)
#     ('waste', -1.1684451288279047)
#     ('awful', -1.0277001734353677)
#     ('poorly', -0.8748317895742782)
#     ('boring', -0.8587249740682945)