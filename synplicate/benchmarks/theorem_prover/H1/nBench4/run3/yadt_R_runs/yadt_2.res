YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 38.4615% of 52 cases. 

MISCLASSIFICATION on training: 1.92308% of 52 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	32	0	0
0	1	19	1

MISCLASSIFICATION on test file: 12.7889% of 649 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	384	53	0
0	30	182	1

DECISION TREE (size=11)

F1 <= -0.550000 : 0 (18.0/1.3)
F1 > -0.550000 :
|   F10 <= 1.600000 : 0 (9.0/1.3)
|   F10 > 1.600000 :
|   |   F10 <= 3.400000 :
|   |   |   F1 <= 0.290000 : 0 (2.0/1.0)
|   |   |   F1 > 0.290000 :
|   |   |   |   F1 <= 0.990000 : 1 (7.0/1.3)
|   |   |   |   F1 > 0.990000 : 0 (4.0/2.2)
|   |   F10 > 3.400000 : 1 (12.0/1.3)
