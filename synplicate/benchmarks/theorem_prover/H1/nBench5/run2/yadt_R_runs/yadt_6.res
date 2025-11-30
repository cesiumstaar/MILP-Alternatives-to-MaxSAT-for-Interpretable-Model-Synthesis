YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 1 (2 classes)
MAJORITY MISCLASSIFICATION on training: 47.4138% of 116 cases. 

MISCLASSIFICATION on training: 0.862069% of 116 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	55	0	0
0	1	60	1

MISCLASSIFICATION on test file: 19.3548% of 403 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	167	69	0
0	9	158	1

DECISION TREE (size=13)

F1 <= 0.220000 :
|   F10 <= 4.800000 : 0 (34.0/1.4)
|   F10 > 4.800000 :
|   |   F1 <= -0.380000 : 0 (2.0/1.0)
|   |   F1 > -0.380000 : 1 (4.0/1.2)
F1 > 0.220000 :
|   F10 <= 2.900000 :
|   |   F10 <= 1.500000 : 0 (12.0/1.3)
|   |   F10 > 1.500000 :
|   |   |   F1 <= 1.430000 : 1 (12.0/1.3)
|   |   |   F1 > 1.430000 : 0 (8.0/2.4)
|   F10 > 2.900000 : 1 (44.0/1.4)
