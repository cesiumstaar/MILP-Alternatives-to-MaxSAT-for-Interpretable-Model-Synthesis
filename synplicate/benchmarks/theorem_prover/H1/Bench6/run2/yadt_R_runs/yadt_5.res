YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 42% of 100 cases. 

MISCLASSIFICATION on training: 0% of 100 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	58	0	0
0	0	42	1

MISCLASSIFICATION on test file: 11.4362% of 752 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	451	66	0
0	20	215	1

DECISION TREE (size=17)

F1 <= -0.260000 : 0 (33.0/1.4)
F1 > -0.260000 :
|   F10 <= 1.600000 : 0 (12.0/1.3)
|   F10 > 1.600000 :
|   |   F1 <= 1.330000 :
|   |   |   F1 <= 0.000000 :
|   |   |   |   F10 <= 3.900000 : 0 (2.0/1.0)
|   |   |   |   F10 > 3.900000 : 1 (3.0/1.1)
|   |   |   F1 > 0.000000 : 1 (29.0/1.4)
|   |   F1 > 1.330000 :
|   |   |   F10 <= 4.400000 :
|   |   |   |   F1 <= 2.000000 :
|   |   |   |   |   F10 <= 3.000000 : 0 (3.0/1.1)
|   |   |   |   |   F10 > 3.000000 : 1 (3.0/1.1)
|   |   |   |   F1 > 2.000000 : 0 (8.0/1.3)
|   |   |   F10 > 4.400000 : 1 (7.0/1.3)
