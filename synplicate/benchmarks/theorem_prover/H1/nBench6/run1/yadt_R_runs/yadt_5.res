YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 40% of 100 cases. 

MISCLASSIFICATION on training: 2% of 100 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	59	1	0
0	1	39	1

MISCLASSIFICATION on test file: 14.4686% of 781 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	447	96	0
0	17	221	1

DECISION TREE (size=15)

F1 <= 0.110000 : 0 (40.0/2.6)
F1 > 0.110000 :
|   F10 <= 2.500000 :
|   |   F10 <= 1.500000 : 0 (12.0/1.3)
|   |   F10 > 1.500000 :
|   |   |   F1 <= 1.300000 :
|   |   |   |   F1 <= 0.450000 : 0 (2.0/1.0)
|   |   |   |   F1 > 0.450000 : 1 (3.0/1.1)
|   |   |   F1 > 1.300000 : 0 (3.0/1.1)
|   F10 > 2.500000 :
|   |   F1 <= 1.910000 : 1 (30.0/1.4)
|   |   F1 > 1.910000 :
|   |   |   F10 <= 3.400000 : 0 (3.0/1.1)
|   |   |   F10 > 3.400000 : 1 (7.0/2.4)
