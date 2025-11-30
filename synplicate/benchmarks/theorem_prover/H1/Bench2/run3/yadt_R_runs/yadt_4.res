YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 42.1687% of 83 cases. 

MISCLASSIFICATION on training: 0% of 83 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	48	0	0
0	0	35	1

MISCLASSIFICATION on test file: 21.1094% of 649 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	327	125	0
0	12	185	1

DECISION TREE (size=19)

F1 <= -0.230000 :
|   F10 <= 5.200000 : 0 (26.0/1.3)
|   F10 > 5.200000 :
|   |   F1 <= -1.230000 : 0 (2.0/1.0)
|   |   F1 > -1.230000 : 1 (2.0/1.0)
F1 > -0.230000 :
|   F10 <= 2.300000 :
|   |   F10 <= 1.500000 : 0 (10.0/1.3)
|   |   F10 > 1.500000 :
|   |   |   F1 <= 1.000000 :
|   |   |   |   F1 <= 0.460000 : 0 (2.0/1.0)
|   |   |   |   F1 > 0.460000 : 1 (3.0/1.1)
|   |   |   F1 > 1.000000 : 0 (5.0/1.2)
|   F10 > 2.300000 :
|   |   F1 <= 1.770000 : 1 (27.0/1.4)
|   |   F1 > 1.770000 :
|   |   |   F10 <= 4.100000 : 0 (3.0/1.1)
|   |   |   F10 > 4.100000 : 1 (3.0/1.1)
