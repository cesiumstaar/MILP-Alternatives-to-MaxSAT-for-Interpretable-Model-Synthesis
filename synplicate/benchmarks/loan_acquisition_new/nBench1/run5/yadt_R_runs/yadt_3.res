YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 1 (2 classes)
MAJORITY MISCLASSIFICATION on training: 10.2941% of 68 cases. 

MISCLASSIFICATION on training: 1.47059% of 68 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	6	1	0
0	0	61	1

MISCLASSIFICATION on test file: 5.07246% of 276 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	12	14	0
0	0	250	1

DECISION TREE (size=5)

monthly_income <= 4801.000000 : 1 (36.0/1.4)
monthly_income > 4801.000000 :
|   age <= 28 : 0 (6.0/1.2)
|   age > 28 : 1 (26.0/2.6)
