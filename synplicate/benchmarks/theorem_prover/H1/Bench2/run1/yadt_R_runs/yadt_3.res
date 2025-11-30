YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 1 (2 classes)
MAJORITY MISCLASSIFICATION on training: 39.7059% of 68 cases. 

MISCLASSIFICATION on training: 1.47059% of 68 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	26	1	0
0	0	41	1

MISCLASSIFICATION on test file: 27.9321% of 648 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	236	172	0
0	9	231	1

DECISION TREE (size=15)

F1 <= -0.120000 :
|   F10 <= 4.900000 : 0 (14.0/1.3)
|   F10 > 4.900000 :
|   |   F1 <= -1.120000 : 0 (3.0/1.1)
|   |   F1 > -1.120000 : 1 (3.0/1.1)
F1 > -0.120000 :
|   F10 <= 2.700000 :
|   |   F1 <= 0.920000 : 1 (7.0/2.4)
|   |   F1 > 0.920000 : 0 (7.0/1.3)
|   F10 > 2.700000 :
|   |   F1 <= 1.880000 : 1 (27.0/1.4)
|   |   F1 > 1.880000 :
|   |   |   F10 <= 4.000000 : 0 (2.0/1.0)
|   |   |   F10 > 4.000000 : 1 (5.0/1.2)
