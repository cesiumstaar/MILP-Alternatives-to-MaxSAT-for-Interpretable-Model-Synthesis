YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 1 (2 classes)
MAJORITY MISCLASSIFICATION on training: 46.3415% of 164 cases. 

MISCLASSIFICATION on training: 1.82927% of 164 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	76	0	0
0	3	85	1

MISCLASSIFICATION on test file: 19.6498% of 514 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	210	90	0
0	11	203	1

DECISION TREE (size=15)

F1 <= -0.190000 : 0 (33.0/1.4)
F1 > -0.190000 :
|   F10 <= 1.600000 : 0 (20.0/1.3)
|   F10 > 1.600000 :
|   |   F10 <= 3.200000 :
|   |   |   F1 <= 0.310000 : 0 (6.0/1.2)
|   |   |   F1 > 0.310000 :
|   |   |   |   F1 <= 1.430000 : 1 (15.0/1.3)
|   |   |   |   F1 > 1.430000 : 0 (12.0/3.6)
|   |   F10 > 3.200000 :
|   |   |   F1 <= 0.000000 :
|   |   |   |   F10 <= 4.800000 : 0 (8.0/2.4)
|   |   |   |   F10 > 4.800000 : 1 (8.0/1.3)
|   |   |   F1 > 0.000000 : 1 (62.0/1.4)
