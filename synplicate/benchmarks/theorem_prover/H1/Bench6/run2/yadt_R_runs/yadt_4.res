YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 46.4286% of 84 cases. 

MISCLASSIFICATION on training: 0% of 84 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	45	0	0
0	0	39	1

MISCLASSIFICATION on test file: 14.8905% of 685 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	382	87	0
0	15	201	1

DECISION TREE (size=17)

F1 <= -0.260000 : 0 (25.0/1.3)
F1 > -0.260000 :
|   F10 <= 1.600000 : 0 (11.0/1.3)
|   F10 > 1.600000 :
|   |   F1 <= 1.330000 :
|   |   |   F1 <= 0.000000 :
|   |   |   |   F10 <= 3.900000 : 0 (2.0/1.0)
|   |   |   |   F10 > 3.900000 : 1 (3.0/1.1)
|   |   |   F1 > 0.000000 : 1 (29.0/1.4)
|   |   F1 > 1.330000 :
|   |   |   F10 <= 4.400000 :
|   |   |   |   F1 <= 2.000000 :
|   |   |   |   |   F10 <= 2.900000 : 0 (2.0/1.0)
|   |   |   |   |   F10 > 2.900000 : 1 (2.0/1.0)
|   |   |   |   F1 > 2.000000 : 0 (5.0/1.2)
|   |   |   F10 > 4.400000 : 1 (5.0/1.2)
