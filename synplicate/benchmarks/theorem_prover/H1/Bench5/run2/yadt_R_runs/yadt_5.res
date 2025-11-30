YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 37% of 100 cases. 

MISCLASSIFICATION on training: 1% of 100 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	62	1	0
0	0	37	1

MISCLASSIFICATION on test file: 12.5% of 408 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	229	44	0
0	7	128	1

DECISION TREE (size=11)

F1 <= -0.030000 : 0 (30.0/1.4)
F1 > -0.030000 :
|   F10 <= 3.100000 :
|   |   F10 <= 1.600000 : 0 (19.0/1.3)
|   |   F10 > 1.600000 :
|   |   |   F1 <= 1.380000 :
|   |   |   |   F1 <= 0.290000 : 0 (5.0/1.2)
|   |   |   |   F1 > 0.290000 : 1 (15.0/2.5)
|   |   |   F1 > 1.380000 : 0 (8.0/1.3)
|   F10 > 3.100000 : 1 (23.0/1.3)
