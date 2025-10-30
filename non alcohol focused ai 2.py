import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv('final_cocktails.csv')

#changes the 'alcoholic' column to numbers
df['alcoholic'] = df['alcoholic'].apply(lambda x: 1 if x.lower() == 'alcoholic' else 0)

def check_alcohol2(text):
    non_alcohol_words = ['juice', 'soda', 'water', 'syrup', 'cream', 'milk', 'coffee', 'tea',
                         'sugar', 'honey', 'lemonade', 'coconut', 'cola', 'ginger ale', 'ice',
                         'mint', 'sherbet', 'egg', 'fruit', 'peach nectar', 'pineapple', 'orange']
    
    if pd.isnull(text):
        return 1  
    text = text.lower()
    for word in non_alcohol_words:
        if word in text:
            return 0  
    return 1  

df['ai_2_pred'] = df['ingredients'].apply(check_alcohol2)

#evaluation for check_alcohol2
accuracy_2 = accuracy_score(df['alcoholic'], df['ai_2_pred'])
precision_2 = precision_score(df['alcoholic'], df['ai_2_pred'])
recall_2 = recall_score(df['alcoholic'], df['ai_2_pred'])
f1_2 = f1_score(df['alcoholic'], df['ai_2_pred'])

print('check_alcohol2 Evaluation:')
print('Accuracy:', round(accuracy_2, 4))
print('Precision:', round(precision_2, 4))
print('Recall:', round(recall_2, 4))
print('F1 Score:', round(f1_2, 4))
