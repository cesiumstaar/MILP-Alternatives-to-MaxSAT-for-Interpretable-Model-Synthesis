YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 46% of 100 cases. 

MISCLASSIFICATION on training: 2% of 100 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	53	1	0
0	1	45	1

MISCLASSIFICATION on test file: 18.0739% of 758 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	411	106	0
0	31	210	1

DECISION TREE (size=15)

F1 <= -0.140000 : 0 (33.0/1.4)
F1 > -0.140000 :
|   F10 <= 2.700000 :
|   |   F10 <= 1.900000 : 0 (10.0/1.3)
|   |   F10 > 1.900000 :
|   |   |   F1 <= 1.350000 :
|   |   |   |   F1 <= 0.350000 : 0 (2.0/1.0)
|   |   |   |   F1 > 0.350000 : 1 (4.0/1.2)
|   |   |   F1 > 1.350000 : 0 (4.0/1.2)
|   F10 > 2.700000 :
|   |   F1 <= 1.940000 : 1 (36.0/2.6)
|   |   F1 > 1.940000 :
|   |   |   F10 <= 4.500000 : 0 (5.0/2.3)
|   |   |   F10 > 4.500000 : 1 (6.0/1.2)
