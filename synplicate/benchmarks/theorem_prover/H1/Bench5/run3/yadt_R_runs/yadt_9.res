YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 39.6341% of 164 cases. 

MISCLASSIFICATION on training: 0.609756% of 164 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	98	1	0
0	0	65	1

MISCLASSIFICATION on test file: 16.6352% of 529 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	267	82	0
0	6	174	1

DECISION TREE (size=15)

F10 <= 2.400000 :
|   F10 <= 1.400000 : 0 (45.0/1.4)
|   F10 > 1.400000 :
|   |   F1 <= 0.320000 : 0 (17.0/1.3)
|   |   F1 > 0.320000 :
|   |   |   F1 <= 1.240000 : 1 (10.0/1.3)
|   |   |   F1 > 1.240000 : 0 (12.0/1.3)
F10 > 2.400000 :
|   F1 <= -0.030000 :
|   |   F10 <= 4.900000 : 0 (21.0/1.3)
|   |   F10 > 4.900000 :
|   |   |   F1 <= -0.370000 : 0 (3.0/1.1)
|   |   |   F1 > -0.370000 : 1 (2.0/1.0)
|   F1 > -0.030000 : 1 (54.0/2.6)
