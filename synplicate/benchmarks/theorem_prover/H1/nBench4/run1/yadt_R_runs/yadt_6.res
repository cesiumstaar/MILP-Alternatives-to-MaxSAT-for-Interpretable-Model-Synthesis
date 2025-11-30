YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 43.1034% of 116 cases. 

MISCLASSIFICATION on training: 1.72414% of 116 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	65	1	0
0	1	49	1

MISCLASSIFICATION on test file: 11.817% of 787 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	463	74	0
0	19	231	1

DECISION TREE (size=15)

F1 <= -0.300000 : 0 (35.0/1.4)
F1 > -0.300000 :
|   F10 <= 1.400000 : 0 (16.0/1.3)
|   F10 > 1.400000 :
|   |   F10 <= 4.000000 :
|   |   |   F1 <= 1.780000 :
|   |   |   |   F1 <= 0.300000 : 0 (4.0/1.2)
|   |   |   |   F1 > 0.300000 :
|   |   |   |   |   F10 <= 2.400000 :
|   |   |   |   |   |   F1 <= 0.850000 : 1 (7.0/1.3)
|   |   |   |   |   |   F1 > 0.850000 : 0 (3.0/1.1)
|   |   |   |   |   F10 > 2.400000 : 1 (15.0/1.3)
|   |   |   F1 > 1.780000 : 0 (8.0/2.4)
|   |   F10 > 4.000000 : 1 (28.0/2.6)
