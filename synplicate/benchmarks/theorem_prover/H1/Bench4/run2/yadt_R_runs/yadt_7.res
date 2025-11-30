YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 31.2977% of 131 cases. 

MISCLASSIFICATION on training: 0.763359% of 131 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	90	0	0
0	1	40	1

MISCLASSIFICATION on test file: 12.2807% of 969 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	618	75	0
0	44	232	1

DECISION TREE (size=19)

F1 <= -0.490000 : 0 (45.0/1.4)
F1 > -0.490000 :
|   F10 <= 2.500000 :
|   |   F10 <= 1.500000 : 0 (23.0/1.3)
|   |   F10 > 1.500000 :
|   |   |   F1 <= 0.980000 :
|   |   |   |   F1 <= 0.330000 : 0 (4.0/1.2)
|   |   |   |   F1 > 0.330000 : 1 (5.0/1.2)
|   |   |   F1 > 0.980000 : 0 (8.0/1.3)
|   F10 > 2.500000 :
|   |   F1 <= 1.950000 :
|   |   |   F1 <= 0.150000 :
|   |   |   |   F10 <= 3.500000 : 0 (3.0/1.1)
|   |   |   |   F10 > 3.500000 : 1 (5.0/1.2)
|   |   |   F1 > 0.150000 : 1 (27.0/1.4)
|   |   F1 > 1.950000 :
|   |   |   F10 <= 4.500000 : 0 (8.0/2.4)
|   |   |   F10 > 4.500000 : 1 (3.0/1.1)
