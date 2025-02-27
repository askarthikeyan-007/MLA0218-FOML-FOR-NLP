def find_s(examples, target_attribute):
    hypothesis = examples[0][:]  
    
    for example in examples:
        if example[target_attribute] == "Yes":  
            for i in range(len(hypothesis)):
                
                if hypothesis[i] != example[i]:
                    hypothesis[i] = '?'  
    
    return hypothesis
examples = [
    ['sunny', 'warm', 'high', 'Yes'],
    ['sunny', 'warm', 'low', 'No'],
    ['sunny', 'cold', 'high', 'Yes'],
    ['rainy', 'warm', 'high', 'No']
]


target_attribute = 3

hypothesis = find_s(examples, target_attribute)

print("Most Specific Hypothesis:", hypothesis)
