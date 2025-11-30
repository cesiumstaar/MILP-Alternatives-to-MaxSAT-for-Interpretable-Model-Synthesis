YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 36.9048% of 84 cases. 

MISCLASSIFICATION on training: 4.7619% of 84 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	53	0	0
0	4	27	1

MISCLASSIFICATION on test file: 14.8359% of 701 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	432	57	0
0	47	165	1

DECISION TREE (size=11)

F10 <= 2.000000 : 0 (40.0/4.9)
F10 > 2.000000 :
|   F1 <= -0.250000 :
|   |   F10 <= 5.000000 : 0 (12.0/1.3)
|   |   F10 > 5.000000 :
|   |   |   F1 <= -0.800000 : 0 (2.0/1.0)
|   |   |   F1 > -0.800000 : 1 (2.0/1.0)
|   F1 > -0.250000 :
|   |   F1 <= 1.750000 : 1 (25.0/1.3)
|   |   F1 > 1.750000 : 0 (3.0/2.1)
