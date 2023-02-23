########################################
#                                      #
#   HW1:  Rabbit Farm                  #
#                                      #
########################################

import re


input = open('C:/Users/USER/Desktop/reinforcement_learning/HW/hw1/rabbit_farm.txt')
output = open('C:/Users/USER/Desktop/reinforcement_learning/HW/hw1/rabbits.out.txt','w')

str_read = input.read()
str_replace = str_read.replace(":", "\n",)
str_replace = str_replace.strip()
list_1 = str_replace.split('\n')

#print(list_1[0:15])             
#print(len(list_1))                         ----->  30905


numbers = [int(temp)for temp in list_1 if temp.isdigit()]

#print(numbers)    check all numbers appear in txt
#print(len(numbers))   check how many numbers in it 
    
# Sort the list---numbers--- from biggest to smallest
numbers.sort(reverse=True)
#print(numbers)

# Adds all the money together to get total money
sum1 = sum(numbers)


#Searching all rabbits & rabbots
result = []
result_2 =[]

for temp in list_1:
    result.append(''.join(re.findall(r'[a-z]', temp)))

#print(result[0:30])

for temp in result:
    ls = list(temp)
    ls.sort()

    result_2.append(''.join(ls))

#print(result_2[0:30])

rabbit_count = result_2.count('abbirt')
rabbot_count = result_2.count('abbort')
#print(rabbit_count)
#print(rabbot_count)

animals = {'rabbit': rabbit_count, 'rabbot': rabbot_count}

output.write(str(numbers) + '\n')
output.write(str(sum1) + '\n')
output.write(str(animals))
output.close()