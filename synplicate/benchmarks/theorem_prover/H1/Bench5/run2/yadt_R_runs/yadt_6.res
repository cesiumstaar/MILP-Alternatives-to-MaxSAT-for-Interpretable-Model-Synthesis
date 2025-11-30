YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 38.7931% of 116 cases. 

MISCLASSIFICATION on training: 0.862069% of 116 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	70	1	0
0	0	45	1

MISCLASSIFICATION on test file: 12.2222% of 450 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	249	48	0
0	7	146	1

DECISION TREE (size=11)

F1 <= -0.030000 : 0 (36.0/1.4)
F1 > -0.030000 :
|   F10 <= 3.100000 :
|   |   F10 <= 1.600000 : 0 (21.0/1.3)
|   |   F10 > 1.600000 :
|   |   |   F1 <= 1.380000 :
|   |   |   |   F1 <= 0.290000 : 0 (5.0/1.2)
|   |   |   |   F1 > 0.290000 : 1 (16.0/2.5)
|   |   |   F1 > 1.380000 : 0 (8.0/1.3)
|   F10 > 3.100000 : 1 (30.0/1.4)
