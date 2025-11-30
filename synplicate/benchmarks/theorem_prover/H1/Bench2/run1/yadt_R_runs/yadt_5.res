YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 48% of 100 cases. 

MISCLASSIFICATION on training: 3% of 100 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	51	1	0
0	2	46	1

MISCLASSIFICATION on test file: 15.3944% of 786 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	419	100	0
0	21	246	1

DECISION TREE (size=15)

F1 <= -0.140000 : 0 (33.0/2.6)
F1 > -0.140000 :
|   F10 <= 2.700000 :
|   |   F1 <= 0.920000 :
|   |   |   F1 <= 0.400000 : 0 (2.0/1.0)
|   |   |   F1 > 0.400000 : 1 (8.0/2.4)
|   |   F1 > 0.920000 : 0 (11.0/1.3)
|   F10 > 2.700000 :
|   |   F10 <= 4.600000 :
|   |   |   F1 <= 1.880000 :
|   |   |   |   F1 <= -0.120000 : 0 (2.0/1.0)
|   |   |   |   F1 > -0.120000 : 1 (18.0/1.3)
|   |   |   F1 > 1.880000 : 0 (5.0/2.3)
|   |   F10 > 4.600000 : 1 (21.0/1.3)
