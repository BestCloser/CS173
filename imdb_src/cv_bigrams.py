import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('sorted.csv')

cv = CountVectorizer(ngram_range=(2,2))
cv.fit(df['review'])
X = cv.transform(df['review'])
print(X.shape) #dimensions of the vectorizer table


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

Y = [0 if i < 25000 else 1 for i in range(50000)] ## 0 for negative, 1 for positive

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = .75)
## We can customize train_size OR test_size

print("cv with bigrams only:")
lr = LogisticRegression(solver='saga', max_iter=10000) #solver specified to remove FutureWarning
lr.fit(X_train, Y_train)
print("Accuracy: %s" % accuracy_score(Y_test, lr.predict(X_test)))



#display top 5 positive and negative words
feature_to_coef = {
    word: coef for word, coef in zip(
        cv.get_feature_names(), lr.coef_[0]
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
