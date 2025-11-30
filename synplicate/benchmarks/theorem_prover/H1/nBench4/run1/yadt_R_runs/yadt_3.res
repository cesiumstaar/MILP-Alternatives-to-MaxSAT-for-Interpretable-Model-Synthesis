YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 1 (2 classes)
MAJORITY MISCLASSIFICATION on training: 38.2353% of 68 cases. 

MISCLASSIFICATION on training: 2.94118% of 68 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	26	0	0
0	2	40	1

MISCLASSIFICATION on test file: 22.6917% of 639 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	291	131	0
0	14	203	1

DECISION TREE (size=13)

F1 <= 0.150000 :
|   F1 <= -0.580000 : 0 (15.0/1.3)
|   F1 > -0.580000 :
|   |   F10 <= 5.000000 : 0 (4.0/1.2)
|   |   F10 > 5.000000 : 1 (2.0/1.0)
F1 > 0.150000 :
|   F10 <= 1.800000 : 0 (5.0/2.3)
|   F10 > 1.800000 :
|   |   F1 <= 1.780000 : 1 (32.0/1.4)
|   |   F1 > 1.780000 :
|   |   |   F10 <= 4.500000 : 0 (4.0/2.2)
|   |   |   F10 > 4.500000 : 1 (6.0/1.2)
