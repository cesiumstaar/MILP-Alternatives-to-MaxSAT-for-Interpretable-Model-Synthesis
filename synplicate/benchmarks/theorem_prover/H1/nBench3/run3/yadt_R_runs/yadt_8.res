YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 47.2973% of 148 cases. 

MISCLASSIFICATION on training: 2.02703% of 148 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	76	2	0
0	1	69	1

MISCLASSIFICATION on test file: 18.6408% of 515 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	239	85	0
0	11	180	1

DECISION TREE (size=13)

F1 <= 0.110000 : 0 (51.0/2.6)
F1 > 0.110000 :
|   F10 <= 2.900000 :
|   |   F1 <= 1.540000 :
|   |   |   F10 <= 1.300000 : 0 (7.0/1.3)
|   |   |   F10 > 1.300000 :
|   |   |   |   F10 <= 1.900000 :
|   |   |   |   |   F1 <= 0.780000 : 1 (3.0/1.1)
|   |   |   |   |   F1 > 0.780000 : 0 (3.0/1.1)
|   |   |   |   F10 > 1.900000 : 1 (14.0/2.5)
|   |   F1 > 1.540000 : 0 (16.0/1.3)
|   F10 > 2.900000 : 1 (54.0/2.6)
