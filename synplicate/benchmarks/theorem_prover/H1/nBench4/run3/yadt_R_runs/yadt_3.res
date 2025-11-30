YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 39.7059% of 68 cases. 

MISCLASSIFICATION on training: 1.47059% of 68 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	40	1	0
0	0	27	1

MISCLASSIFICATION on test file: 14.7937% of 703 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	382	87	0
0	17	217	1

DECISION TREE (size=11)

F10 <= 1.600000 : 0 (23.0/1.3)
F10 > 1.600000 :
|   F1 <= -0.550000 : 0 (12.0/1.3)
|   F1 > -0.550000 :
|   |   F10 <= 3.400000 :
|   |   |   F1 <= 1.450000 :
|   |   |   |   F1 <= 0.290000 : 0 (2.0/1.0)
|   |   |   |   F1 > 0.290000 : 1 (10.0/2.4)
|   |   |   F1 > 1.450000 : 0 (3.0/1.1)
|   |   F10 > 3.400000 : 1 (18.0/1.3)
