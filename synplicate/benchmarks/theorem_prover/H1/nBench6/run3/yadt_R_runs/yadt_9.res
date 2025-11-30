YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 35.5828% of 163 cases. 

MISCLASSIFICATION on training: 0.613497% of 163 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	104	1	0
0	0	58	1

MISCLASSIFICATION on test file: 11.8026% of 932 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	572	96	0
0	14	250	1

DECISION TREE (size=19)

F1 <= -0.280000 : 0 (55.0/1.4)
F1 > -0.280000 :
|   F10 <= 2.600000 :
|   |   F1 <= 1.190000 :
|   |   |   F10 <= 1.200000 : 0 (10.0/1.3)
|   |   |   F10 > 1.200000 :
|   |   |   |   F1 <= 0.280000 : 0 (2.0/1.0)
|   |   |   |   F1 > 0.280000 : 1 (11.0/1.3)
|   |   F1 > 1.190000 : 0 (29.0/1.4)
|   F10 > 2.600000 :
|   |   F1 <= 1.890000 :
|   |   |   F1 <= -0.110000 :
|   |   |   |   F10 <= 4.600000 : 0 (2.0/1.0)
|   |   |   |   F10 > 4.600000 : 1 (2.0/1.0)
|   |   |   F1 > -0.110000 : 1 (38.0/1.4)
|   |   F1 > 1.890000 :
|   |   |   F10 <= 3.700000 : 0 (6.0/1.2)
|   |   |   F10 > 3.700000 : 1 (8.0/2.4)
