import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv('final_cocktails.csv')

#changes the 'alcoholic' column to numbers
df['alcoholic'] = df['alcoholic'].apply(lambda x: 1 if x.lower() == 'alcoholic' else 0)

def check_alcohol1(text):
    alcohol_words = ['vodka', 'rum', 'gin', 'whiskey', 'tequila', 'scotch', 'bourbon',
                     'brandy', 'cognac', 'vermouth', 'liqueur', 'amaretto', 'baileys',
                     'kahlua', 'triple sec', 'grand marnier', 'schnapps', 'applejack',
                     'chartreuse', 'sambuca', 'midori', 'frangelico', 'campari']
    
    if pd.isnull(text):
        return 0
    text = text.lower()
    for word in alcohol_words:
        if word in text:
            return 1
    return 0
#predicts based on ingredients
df['ai_1_pred'] = df['ingredients'].apply(check_alcohol1)

#evaluation for check_alcohol1
accuracy_1 = accuracy_score(df['alcoholic'], df['ai_1_pred'])
precision_1 = precision_score(df['alcoholic'], df['ai_1_pred'])
recall_1 = recall_score(df['alcoholic'], df['ai_1_pred'])
f1_1 = f1_score(df['alcoholic'], df['ai_1_pred'])

print('check_alcohol1 Evaluation:')
print('Accuracy:', round(accuracy_1, 4))
print('Precision:', round(precision_1, 4))
print('Recall:', round(recall_1, 4))
print('F1 Score:', round(f1_1, 4))
