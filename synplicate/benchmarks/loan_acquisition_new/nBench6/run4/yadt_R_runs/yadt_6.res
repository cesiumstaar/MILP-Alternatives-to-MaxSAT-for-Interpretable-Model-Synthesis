YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 1 (2 classes)
MAJORITY MISCLASSIFICATION on training: 2.58621% of 116 cases. 

MISCLASSIFICATION on training: 0% of 116 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	3	0	0
0	0	113	1

MISCLASSIFICATION on test file: 7.32044% of 724 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	50	33	0
0	20	621	1

DECISION TREE (size=5)

age <= 34 :
|   age <= 29 : 0 (3.0/1.1)
|   age > 29 : 1 (6.0/1.2)
age > 34 : 1 (107.0/1.4)
