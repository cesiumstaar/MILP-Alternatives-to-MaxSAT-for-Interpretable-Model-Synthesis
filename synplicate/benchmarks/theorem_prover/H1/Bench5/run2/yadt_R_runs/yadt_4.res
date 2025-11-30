YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 32.1429% of 84 cases. 

MISCLASSIFICATION on training: 1.19048% of 84 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	56	1	0
0	0	27	1

MISCLASSIFICATION on test file: 10.8333% of 360 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	215	33	0
0	6	106	1

DECISION TREE (size=11)

F10 <= 1.600000 : 0 (28.0/1.4)
F10 > 1.600000 :
|   F1 <= -0.030000 : 0 (15.0/1.3)
|   F1 > -0.030000 :
|   |   F10 <= 3.100000 :
|   |   |   F1 <= 1.220000 :
|   |   |   |   F1 <= 0.290000 : 0 (5.0/1.2)
|   |   |   |   F1 > 0.290000 : 1 (13.0/2.5)
|   |   |   F1 > 1.220000 : 0 (8.0/1.3)
|   |   F10 > 3.100000 : 1 (15.0/1.3)
