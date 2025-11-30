YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 1 (2 classes)
MAJORITY MISCLASSIFICATION on training: 10% of 20 cases. 

MISCLASSIFICATION on training: 0% of 20 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	2	0	0
0	0	18	1

MISCLASSIFICATION on test file: 5% of 180 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	12	0	0
0	9	159	1

DECISION TREE (size=5)

age <= 34 :
|   monthly_income <= 4012.000000 : 1 (3.0/1.1)
|   monthly_income > 4012.000000 : 0 (2.0/1.0)
age > 34 : 1 (15.0/1.3)
