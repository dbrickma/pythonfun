
# coding: utf-8

# In[82]:

def ran_check(num1, num2):
     for num in range(num2):
        def SD_Check(num2): 
            for num in range(num2):
                r = range(num2)
                z = sum(range(num2))
                l = len(range(num2))
                av = float(z/l)
                SS = (sum(((av - i)**2 for i in r)))/l
                SD = SS**.5
                return SD
            return SD_Check()[0]
        print SD_Check(num2)
        x = sum(range(num2))
        y = len(range(num2))
        z = x/y
        if num1 >= z + SD_Check(num2): 
            print 'high' 
        elif num1 <= z - SD_Check(num2): 
            print 'low' 
        else: 
            print 'average'
              


# In[83]:

ran_check(4,(240))


# In[26]:

def funthing(num1,num2):
    if num1 > (sum((range(num2)))/len(num2)) + sum(num2):
                print "fun"


# In[24]:




# In[ ]:



