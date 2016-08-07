
# coding: utf-8

# In[6]:

for x in range(101): 
    while x % 3 == 0:
        print "Fizz"
        break 
    while x % 5 == 0: 
        print "Buzz"
        break
    while x % 3 == 0 and x % 5 == 0:
        print "FizzBuzz"
        break 
    


# In[ ]:



