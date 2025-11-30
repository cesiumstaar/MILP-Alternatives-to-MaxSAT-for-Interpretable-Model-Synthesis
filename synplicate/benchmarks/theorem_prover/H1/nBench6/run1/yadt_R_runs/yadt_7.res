YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 37.1212% of 132 cases. 

MISCLASSIFICATION on training: 1.51515% of 132 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	82	1	0
0	1	48	1

MISCLASSIFICATION on test file: 12.7784% of 853 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	508	89	0
0	20	236	1

DECISION TREE (size=15)

F1 <= 0.110000 : 0 (52.0/2.6)
F1 > 0.110000 :
|   F10 <= 2.400000 :
|   |   F10 <= 1.500000 : 0 (19.0/1.3)
|   |   F10 > 1.500000 :
|   |   |   F1 <= 1.190000 :
|   |   |   |   F1 <= 0.450000 : 0 (3.0/1.1)
|   |   |   |   F1 > 0.450000 : 1 (3.0/1.1)
|   |   |   F1 > 1.190000 : 0 (5.0/1.2)
|   F10 > 2.400000 :
|   |   F1 <= 1.910000 : 1 (37.0/1.4)
|   |   F1 > 1.910000 :
|   |   |   F10 <= 3.500000 : 0 (4.0/1.2)
|   |   |   F10 > 3.500000 : 1 (9.0/2.4)
