YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 1 (2 classes)
MAJORITY MISCLASSIFICATION on training: 15% of 20 cases. 

MISCLASSIFICATION on training: 0% of 20 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	3	0	0
0	0	17	1

MISCLASSIFICATION on test file: 8.2% of 500 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	50	30	0
0	11	409	1

DECISION TREE (size=5)

age <= 32 :
|   monthly_income <= 5269.000000 : 1 (3.0/1.1)
|   monthly_income > 5269.000000 : 0 (3.0/1.1)
age > 32 : 1 (14.0/1.3)
