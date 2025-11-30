YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 40.5172% of 116 cases. 

MISCLASSIFICATION on training: 1.72414% of 116 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	69	0	0
0	2	45	1

MISCLASSIFICATION on test file: 14.218% of 422 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	233	52	0
0	8	129	1

DECISION TREE (size=15)

F10 <= 1.600000 : 0 (32.0/1.4)
F10 > 1.600000 :
|   F1 <= 0.170000 : 0 (29.0/3.8)
|   F1 > 0.170000 :
|   |   F10 <= 2.800000 :
|   |   |   F1 <= 1.410000 :
|   |   |   |   F1 <= 0.280000 : 0 (3.0/1.1)
|   |   |   |   F1 > 0.280000 :
|   |   |   |   |   F10 <= 2.100000 :
|   |   |   |   |   |   F1 <= 1.000000 : 1 (3.0/1.1)
|   |   |   |   |   |   F1 > 1.000000 : 0 (2.0/1.0)
|   |   |   |   |   F10 > 2.100000 : 1 (8.0/1.3)
|   |   |   F1 > 1.410000 : 0 (5.0/1.2)
|   |   F10 > 2.800000 : 1 (34.0/1.4)
