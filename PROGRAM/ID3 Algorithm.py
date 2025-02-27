import math
import pandas as pd

def entropy(data):
    target = data.columns[-1]
    class_counts = data[target].value_counts()
    entropy_value = 0
    for count in class_counts:
        prob = count / len(data)
        entropy_value -= prob * math.log2(prob)
    return entropy_value

def information_gain(data, feature):
    target = data.columns[-1]
    total_entropy = entropy(data)
    feature_values = data[feature].unique()
    
    weighted_entropy = 0
    for value in feature_values:
        subset = data[data[feature] == value]
        weighted_entropy += (len(subset) / len(data)) * entropy(subset)
    
    return total_entropy - weighted_entropy

def id3(data):
    target = data.columns[-1]
    
    if len(data[target].unique()) == 1:
        return data[target].iloc[0]
    
    if len(data.columns) == 1:
        return data[target].mode()[0]
    
    best_feature = max(data.columns[:-1], key=lambda feature: information_gain(data, feature))
    
    tree = {best_feature: {}}
    for value in data[best_feature].unique():
        subset = data[data[best_feature] == value]
        tree[best_feature][value] = id3(subset.drop(columns=[best_feature]))
    
    return tree

data = pd.DataFrame({
    'Sky': ['Sunny', 'Sunny', 'Rainy', 'Sunny', 'Rainy'],
    'Temperature': ['Warm', 'Warm', 'Warm', 'Cold', 'Warm'],
    'Humidity': ['High', 'High', 'High', 'High', 'Low'],
    'Wind': ['Strong', 'Weak', 'Strong', 'Weak', 'Weak'],
    'PlayTennis': ['Yes', 'Yes', 'No', 'Yes', 'No']
})

print("Decision Tree:", id3(data))
