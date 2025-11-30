YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 44.2308% of 52 cases. 

MISCLASSIFICATION on training: 0% of 52 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	29	0	0
0	0	23	1

MISCLASSIFICATION on test file: 11.8519% of 270 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	144	28	0
0	4	94	1

DECISION TREE (size=11)

F10 <= 1.600000 : 0 (12.0/1.3)
F10 > 1.600000 :
|   F1 <= -0.040000 : 0 (9.0/1.3)
|   F1 > -0.040000 :
|   |   F10 <= 3.100000 :
|   |   |   F1 <= 0.390000 : 0 (4.0/1.2)
|   |   |   F1 > 0.390000 :
|   |   |   |   F1 <= 1.170000 : 1 (8.0/1.3)
|   |   |   |   F1 > 1.170000 : 0 (4.0/1.2)
|   |   F10 > 3.100000 : 1 (15.0/1.3)
