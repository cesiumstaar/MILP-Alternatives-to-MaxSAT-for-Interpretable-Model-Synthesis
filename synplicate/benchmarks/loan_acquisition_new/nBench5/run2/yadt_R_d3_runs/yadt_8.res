YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 1 (2 classes)
MAJORITY MISCLASSIFICATION on training: 14.8649% of 148 cases. 

MISCLASSIFICATION on training: 0.675676% of 148 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	22	0	0
0	1	125	1

MISCLASSIFICATION on test file: 3.89908% of 436 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	54	10	0
0	7	365	1

DECISION TREE (size=7)

age <= 27 :
|   monthly_income <= 3807.000000 :
|   |   age <= 25 : 0 (7.0/2.4)
|   |   age > 25 : 1 (3.0/1.1)
|   monthly_income > 3807.000000 : 0 (16.0/1.3)
age > 27 : 1 (122.0/1.4)
