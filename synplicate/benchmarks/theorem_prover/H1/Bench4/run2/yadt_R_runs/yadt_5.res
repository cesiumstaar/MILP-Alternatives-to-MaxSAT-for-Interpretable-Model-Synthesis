YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 30.303% of 99 cases. 

MISCLASSIFICATION on training: 4.0404% of 99 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	67	2	0
0	2	28	1

MISCLASSIFICATION on test file: 13.9563% of 824 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	515	80	0
0	35	194	1

DECISION TREE (size=13)

F1 <= -0.490000 : 0 (33.0/1.4)
F1 > -0.490000 :
|   F10 <= 2.000000 : 0 (28.0/3.7)
|   F10 > 2.000000 :
|   |   F1 <= 1.950000 :
|   |   |   F10 <= 3.500000 :
|   |   |   |   F1 <= 0.150000 : 0 (3.0/1.1)
|   |   |   |   F1 > 0.150000 : 1 (11.0/2.5)
|   |   |   F10 > 3.500000 : 1 (16.0/1.3)
|   |   F1 > 1.950000 :
|   |   |   F10 <= 3.800000 : 0 (5.0/1.2)
|   |   |   F10 > 3.800000 : 1 (3.0/2.1)
