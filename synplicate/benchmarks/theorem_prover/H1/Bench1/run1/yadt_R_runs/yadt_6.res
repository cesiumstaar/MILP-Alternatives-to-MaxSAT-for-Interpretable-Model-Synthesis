YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 44.8276% of 116 cases. 

MISCLASSIFICATION on training: 0.862069% of 116 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	63	1	0
0	0	52	1

MISCLASSIFICATION on test file: 18.2254% of 417 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	187	72	0
0	4	154	1

DECISION TREE (size=11)

F1 <= 0.100000 : 0 (37.0/1.4)
F1 > 0.100000 :
|   F10 <= 2.600000 :
|   |   F1 <= 0.980000 :
|   |   |   F10 <= 1.400000 : 0 (7.0/1.3)
|   |   |   F10 > 1.400000 :
|   |   |   |   F1 <= 0.330000 : 0 (3.0/1.1)
|   |   |   |   F1 > 0.330000 : 1 (8.0/1.3)
|   |   F1 > 0.980000 : 0 (16.0/1.3)
|   F10 > 2.600000 : 1 (45.0/2.6)
