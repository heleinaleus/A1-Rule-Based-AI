import pandas as pd
import matplotlib.pyplot as plt

data = {
    'Model': ['Alcohol-Focused AI', 'Non-Alcohol Focused AI'],
    'Accuracy': [0.93, 0.91],
    'Precision': [0.95, 0.89],
    'Recall': [0.88, 0.94],
    'F1 Score': [0.91, 0.91]}

df = pd.DataFrame(data) #turns the dictionary into a table 

#makes the bar graph compare the model's scores
df.plot.bar(x='Model', y=['Accuracy', 'Precision', 'Recall', 'F1 Score'])   

plt.title('Performance Comparison of Rule-Based AI Models')
plt.ylabel('Score')
plt.ylim(0, 1)

plt.show()


