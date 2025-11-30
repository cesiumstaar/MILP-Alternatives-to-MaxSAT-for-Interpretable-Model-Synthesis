YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 33.7838% of 148 cases. 

MISCLASSIFICATION on training: 1.35135% of 148 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	97	1	0
0	1	49	1

MISCLASSIFICATION on test file: 13.563% of 929 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	554	105	0
0	21	249	1

DECISION TREE (size=15)

F1 <= 0.110000 : 0 (60.0/2.6)
F1 > 0.110000 :
|   F10 <= 2.500000 :
|   |   F10 <= 1.500000 : 0 (23.0/1.3)
|   |   F10 > 1.500000 :
|   |   |   F1 <= 1.190000 :
|   |   |   |   F1 <= 0.450000 : 0 (3.0/1.1)
|   |   |   |   F1 > 0.450000 : 1 (4.0/1.2)
|   |   |   F1 > 1.190000 : 0 (7.0/1.3)
|   F10 > 2.500000 :
|   |   F1 <= 1.910000 : 1 (37.0/1.4)
|   |   F1 > 1.910000 :
|   |   |   F10 <= 3.500000 : 0 (5.0/1.2)
|   |   |   F10 > 3.500000 : 1 (9.0/2.4)
