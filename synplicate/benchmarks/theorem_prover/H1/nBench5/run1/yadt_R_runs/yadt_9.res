YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 39.6341% of 164 cases. 

MISCLASSIFICATION on training: 1.82927% of 164 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	97	2	0
0	1	64	1

MISCLASSIFICATION on test file: 15.6364% of 550 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	299	83	0
0	3	165	1

DECISION TREE (size=17)

F10 <= 1.600000 : 0 (43.0/1.4)
F10 > 1.600000 :
|   F1 <= 0.170000 :
|   |   F10 <= 4.900000 : 0 (40.0/2.6)
|   |   F10 > 4.900000 : 1 (4.0/2.2)
|   F1 > 0.170000 :
|   |   F10 <= 2.900000 :
|   |   |   F1 <= 1.410000 :
|   |   |   |   F1 <= 0.280000 : 0 (3.0/1.1)
|   |   |   |   F1 > 0.280000 :
|   |   |   |   |   F10 <= 2.100000 :
|   |   |   |   |   |   F1 <= 1.000000 : 1 (3.0/1.1)
|   |   |   |   |   |   F1 > 1.000000 : 0 (2.0/1.0)
|   |   |   |   |   F10 > 2.100000 : 1 (12.0/1.3)
|   |   |   F1 > 1.410000 : 0 (10.0/1.3)
|   |   F10 > 2.900000 : 1 (47.0/2.6)
