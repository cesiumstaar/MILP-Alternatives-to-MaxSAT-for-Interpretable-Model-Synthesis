YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 37.1622% of 148 cases. 

MISCLASSIFICATION on training: 1.35135% of 148 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	91	2	0
0	0	55	1

MISCLASSIFICATION on test file: 16.1793% of 513 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	281	77	0
0	6	149	1

DECISION TREE (size=13)

F1 <= -0.100000 : 0 (48.0/1.4)
F1 > -0.100000 :
|   F10 <= 2.500000 :
|   |   F1 <= 0.900000 :
|   |   |   F10 <= 2.100000 :
|   |   |   |   F10 <= 1.300000 : 0 (12.0/1.3)
|   |   |   |   F10 > 1.300000 :
|   |   |   |   |   F1 <= 0.440000 : 0 (5.0/1.2)
|   |   |   |   |   F1 > 0.440000 : 1 (5.0/2.3)
|   |   |   F10 > 2.100000 : 1 (6.0/1.2)
|   |   F1 > 0.900000 : 0 (26.0/1.3)
|   F10 > 2.500000 : 1 (46.0/2.6)
