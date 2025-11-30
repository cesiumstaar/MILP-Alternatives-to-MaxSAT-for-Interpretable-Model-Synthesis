YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 45% of 100 cases. 

MISCLASSIFICATION on training: 2% of 100 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	53	2	0
0	0	45	1

MISCLASSIFICATION on test file: 12.78% of 759 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	437	79	0
0	18	225	1

DECISION TREE (size=15)

F1 <= -0.300000 : 0 (33.0/1.4)
F1 > -0.300000 :
|   F10 <= 1.400000 : 0 (10.0/1.3)
|   F10 > 1.400000 :
|   |   F10 <= 4.000000 :
|   |   |   F1 <= 0.300000 : 0 (4.0/1.2)
|   |   |   F1 > 0.300000 :
|   |   |   |   F1 <= 0.850000 : 1 (15.0/1.3)
|   |   |   |   F1 > 0.850000 :
|   |   |   |   |   F1 <= 2.420000 :
|   |   |   |   |   |   F10 <= 2.400000 : 0 (2.0/1.0)
|   |   |   |   |   |   F10 > 2.400000 : 1 (7.0/2.4)
|   |   |   |   |   F1 > 2.420000 : 0 (4.0/1.2)
|   |   F10 > 4.000000 : 1 (25.0/2.5)
