import numpy as np 
import pandas as pd


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df1 = pd.read_csv('train.csv')
df2 = pd.read_csv('test.csv')
sam = pd.read_csv('sample_submission.csv')

sub1 = pd.read_csv('submission_Final_1.csv')

sub2 = pd.read_csv('submission_Final_2.csv')

def generate(main, support, coeff):
    
    g = main.copy()    
    for i in main.columns[1:]:
        
        res = []
        lm, ls = [], []        
        lm = main[i].tolist()
        ls = support[i].tolist()  
        
        for j in range(len(main)):
            res.append((lm[j] * coeff) + (ls[j] * (1.- coeff)))            
        g[i] = res
        
    return g

def generation(main, support, coeff): 
    sub1  = support.copy()
    sub1v = sub1.values    
    
    sub2  = main.copy() 
    sub2v = sub2.values
       
    imp  = main.copy()    
    impv = imp.values  
    NCLASS = 9
    number = 0
    
    for i in range (len(main)):               
        
        row1 = sub1v[i,1:]
        row2 = sub2v[i,1:]
        row1_sort = np.sort(row1)
        row2_sort = np.sort(row2) 
        
        row = (row2 * coeff) + (row1 * (1.0 - coeff))     
        
        for j in range (NCLASS):             
            if ((row2[j] == row2_sort[8]) and (row1[j] != row1_sort[8])):                                
                row = row2 
                number = number + 1            
#               print(number, i)
        
        impv[i, 1:] = row
    
    imp.iloc[:, 1:] = impv[:, 1:]                                     
    return imp
sub = generation(sub2, sub1, 0.40)

sub_ens = sub

sub_ens.to_csv("my_submission.csv",index=False)
# Public Score: 1.74378
