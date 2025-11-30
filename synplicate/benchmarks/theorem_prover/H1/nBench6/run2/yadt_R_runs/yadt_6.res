YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 30.1724% of 116 cases. 

MISCLASSIFICATION on training: 3.44828% of 116 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	80	1	0
0	3	32	1

MISCLASSIFICATION on test file: 12.6002% of 873 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	551	78	0
0	32	212	1

DECISION TREE (size=11)

F10 <= 1.900000 : 0 (53.0/3.8)
F10 > 1.900000 :
|   F1 <= -0.130000 : 0 (20.0/1.3)
|   F1 > -0.130000 :
|   |   F10 <= 3.400000 :
|   |   |   F1 <= 1.350000 :
|   |   |   |   F1 <= 0.300000 : 0 (4.0/2.2)
|   |   |   |   F1 > 0.300000 : 1 (11.0/1.3)
|   |   |   F1 > 1.350000 : 0 (6.0/1.2)
|   |   F10 > 3.400000 : 1 (22.0/2.5)
