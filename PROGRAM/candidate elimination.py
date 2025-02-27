import pandas as pd

def candidate_elimination(data):
    target = data.columns[-1]  
    attributes = data.columns[:-1] 
    
   
    specific_hypothesis = list(data.iloc[0, :-1])  

    
    general_hypothesis = [['?' for _ in range(len(attributes))]] 

    for i in range(1, len(data)):
        if data.iloc[i][target] == 'Yes':  
           
            for j in range(len(specific_hypothesis)):
                if specific_hypothesis[j] != data.iloc[i, j]:
                    specific_hypothesis[j] = '?' 
        else:  
            general_hypothesis = [g for g in general_hypothesis if not is_consistent_with_negative(g, data.iloc[i])]
            
    return specific_hypothesis, general_hypothesis

def is_consistent_with_negative(general_hypothesis, negative_sample):
    """ Helper function to check if a general hypothesis is consistent with a negative sample. """
    for i in range(len(general_hypothesis)):
        if general_hypothesis[i] != '?' and general_hypothesis[i] != negative_sample[i]:
            return False
    return True


data = pd.DataFrame({
    'Sky': ['Sunny', 'Sunny', 'Rainy', 'Sunny', 'Rainy'],
    'Temperature': ['Warm', 'Warm', 'Warm', 'Cold', 'Warm'],
    'Humidity': ['High', 'High', 'High', 'High', 'Low'],
    'Wind': ['Strong', 'Weak', 'Strong', 'Weak', 'Weak'],
    'PlayTennis': ['Yes', 'Yes', 'No', 'Yes', 'No']
})

specific, general = candidate_elimination(data)
print("Specific Hypothesis:", specific)
print("General Hypotheses:", general)
