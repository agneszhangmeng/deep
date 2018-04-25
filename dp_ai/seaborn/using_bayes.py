from seaborn.think_bayes import Pmf
pmf = Pmf()

for x in [1,2,3,4,5,6]:
    pmf.Set(x,1/6.0)


pmf.Set('Bowl1',0.5)
pmf.Set('Bowl2',0.5)

pmf.Mult('Bowl1',0.75)
pmf.Mult('Bowl2',0.5)

pmf.Normalize()
print (pmf.Prob('Bowl1'))