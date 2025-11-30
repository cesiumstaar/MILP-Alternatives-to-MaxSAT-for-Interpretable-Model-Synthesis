YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 36.5854% of 164 cases. 

MISCLASSIFICATION on training: 1.21951% of 164 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	104	0	0
0	2	58	1

MISCLASSIFICATION on test file: 12.772% of 1057 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	634	96	0
0	39	288	1

DECISION TREE (size=13)

F1 <= 0.220000 : 0 (67.0/3.8)
F1 > 0.220000 :
|   F10 <= 1.400000 : 0 (24.0/1.3)
|   F10 > 1.400000 :
|   |   F1 <= 1.380000 : 1 (45.0/1.4)
|   |   F1 > 1.380000 :
|   |   |   F10 <= 3.800000 :
|   |   |   |   F1 <= 1.790000 :
|   |   |   |   |   F10 <= 3.100000 : 0 (3.0/1.1)
|   |   |   |   |   F10 > 3.100000 : 1 (2.0/1.0)
|   |   |   |   F1 > 1.790000 : 0 (12.0/1.3)
|   |   |   F10 > 3.800000 : 1 (11.0/1.3)
