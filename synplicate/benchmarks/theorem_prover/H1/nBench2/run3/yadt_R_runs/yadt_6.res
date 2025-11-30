YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 46.5517% of 116 cases. 

MISCLASSIFICATION on training: 0.862069% of 116 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	62	0	0
0	1	53	1

MISCLASSIFICATION on test file: 12.8916% of 830 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	468	91	0
0	16	255	1

DECISION TREE (size=15)

F1 <= 0.100000 : 0 (42.0/1.4)
F1 > 0.100000 :
|   F10 <= 2.300000 :
|   |   F10 <= 1.200000 : 0 (6.0/1.2)
|   |   F10 > 1.200000 :
|   |   |   F1 <= 1.260000 :
|   |   |   |   F1 <= 0.320000 : 0 (4.0/1.2)
|   |   |   |   F1 > 0.320000 : 1 (5.0/1.2)
|   |   |   F1 > 1.260000 : 0 (5.0/1.2)
|   F10 > 2.300000 :
|   |   F1 <= 1.600000 : 1 (39.0/1.4)
|   |   F1 > 1.600000 :
|   |   |   F10 <= 3.900000 : 0 (6.0/2.3)
|   |   |   F10 > 3.900000 : 1 (9.0/1.3)
