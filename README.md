## 	ANN for regression problemm

https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/

**Basic Perceptron :**
![enter image description here](https://github.com/DevMAdi/Artifical--Neural-Network/blob/master/img/Screenshot%20%282766%29.png?raw=true)

###	Important Things to learn:
	- Neuron				- Activation Function
	- How neural n/w work	- how neural n/w learn
	- Gradient descent		- Stochastic Gradient descent
	- Backpropagation

### Neuron Descp :
![enter image description here](https://github.com/DevMAdi/Artifical--Neural-Network/blob/master/img/Screenshot%20%282767%29.png?raw=true)

 - **i/p Neuron** must be **independent variable**
 - All **independent variable** here are from 1 single row of database, so if we have 1000 records, then signals will go 1000 times in i/p Neuron
 - **Independent Neuron** are features such as age, salary, tenure. etc
 - i/p Neuron/node/var. must be **standardized**.
	> Process of putting different variables on the same scale. This process allows to compare scores between different types of variables
	> resealing one or more attributes so that they have a mean value of 0 and a standard deviation of 1. Standardization assumes that your data has a Gaussian (bell curve) distribution
	
 - Sometimes we **normalize** i/p node instead of standardizing
> part of data preparation to change the values of numeric columns in the dataset to a common scale, without distorting differences in the ranges of values.
> change the values of numeric columns in a dataset to use a common scale when the features in the data have different ranges.
> normalization makes sure that all of your data looks and reads the same way across all records.
> Normalizing the data generally speeds up learning and leads to faster convergence.
> subtract by min value & divide by max-min value by range of value to get value between 0 & 1
## **Normalization v/s standardization**
Normalization usually means to scale a variable to have a values between 0 and 1, while standardization transforms data to have a mean of zero and a standard deviation of 1.
Done so that values are in same range & neural n/w can process fastly.

**For o/p node/Neuron/var:** 	
![enter image description here](https://github.com/DevMAdi/Artifical--Neural-Network/blob/master/img/Screenshot%20%282768%29.png?raw=true)

	can be: continuous	(Price)
			Binary		(Yes/no decision)
			categorical i.e multiple o/p as these will be dummy variable showing our categories.

**For weights/synapse of neuron:** 	

![enter image description here](https://github.com/DevMAdi/Artifical--Neural-Network/blob/master/img/Screenshot%20%282770%29.png?raw=true)
________________________________________________________

 - weights are how neuron n/w learn
 - By adjusting wt. neural n/w decide in every single case what signal is imp. & what signal is not imp. to certain neuron.
 - what signal get passed along & do not pass or to what extent gets passed.
 - They are only things that get adjusted in modal.
 - Gradient descent,Stochastic Grad.descent, Backpropagation help to change wt.
 
**For neuron:** 
![enter image description here](https://github.com/DevMAdi/Artifical--Neural-Network/blob/master/img/Screenshot%20%282771%29.png?raw=true)

 - weighted sum of all value calculated
 - applied to activation Function.
 - passed along to next neuron.

## Activation Function
______________________
1) 	**Threshold Function**
![enter image description here](https://github.com/DevMAdi/Artifical--Neural-Network/blob/master/img/Screenshot%20%282772%29.png?raw=true)
2) 	**Sigmoid Function**
![enter image description here](https://github.com/DevMAdi/Artifical--Neural-Network/blob/master/img/Screenshot%20%282773%29.png?raw=true)
3) 	**Rectifier Function**
![enter image description here](https://github.com/DevMAdi/Artifical--Neural-Network/blob/master/img/Screenshot%20%282775%29.png?raw=true)
4) 	**Hyperboli Tanget Function**
![enter image description here](https://github.com/DevMAdi/Artifical--Neural-Network/blob/master/img/Screenshot%20%282774%29.png?raw=true)

1. can be use for yes/no decisions  i.e depend. var has binary outcome
2. can be use for yes/no decisions. don't have kinks as in (1). 
		useful in o/p layer.
3. has kink. still very useful. value gradually increase after certain point. eg. if building is 10-30 yrs old- low cost due to structural damage etc. if building is 100+ - value increase due to historic imp. & keep increasing as more old it is
			mainly use in hidden layer
4. 	value goes < 0.
### 	How Neuron N/w Work? (assume neural n/e is alrdy trained)
![enter image description here](https://github.com/DevMAdi/Artifical--Neural-Network/blob/master/img/Screenshot%20%282776%29.png?raw=true)
initially all neuron will be interconnected to 1 another
	after training we get this type of modal where 1 neuron will be dominated by weights of few neuron & not all.
Neural n/w finds these strong connection & increase weights of those neuron. Thus accuracy increase

**final modal:**
![enter image description here](https://github.com/DevMAdi/Artifical--Neural-Network/blob/master/img/Screenshot%20%282778%29.png?raw=true)

### How Neural n/w train/learn?
method 1) Hard coding
method 2) create neural n/w & let it figure our after providing it i/p & o/p..

* Things to note :
	1. y- actual o/p value
	2. ŷ- o/p value from n/w
	3. all neuron take i/p from 1 row at time
	   so if we have 1000 row. then 1000 entries into i/p neuron

![enter image description here](https://github.com/DevMAdi/Artifical--Neural-Network/blob/master/img/Screenshot%20%282779%29.png?raw=true)
step 1: i/p(s) given along with wt.
step 2: weighted sum calculated- activation func- we get ŷ now.
step 3: compare ŷ with y. both plotted on chart
step 4: we get cost func as 		

    C= 	1/2(y- ŷ)sq.
**what is cost func?**
> Above cost func. is Mean Squared Error cost function, their are many variants(this 1 is use for grad. descent)
> 
> quantifies the error between predicted values and expected values and presents it in the form of a single real number
> should be low as possible

step 5: cost func. also plotted
step 6: the diff. is then propagated back via neuron & wt. are adjusted.
Aim: 	adjust wt. to reduce cost func.

1 entry at time- 
![enter image description here](https://github.com/DevMAdi/Artifical--Neural-Network/blob/master/img/Screenshot%20%282781%29.png?raw=true)
**1 epoch**= traverse through entire dataset.

For entire dataset:
![enter image description here](https://github.com/DevMAdi/Artifical--Neural-Network/blob/master/img/Screenshot%20%282782%29.png?raw=true)

 - 1st row passed to neuron. we get ŷ & then plotted.
 - 2nd row passed to neuron. we get ŷ & then plotted & so on. for every row
 - once ŷ plotted for all row
 - compare with actual value(y). plotted next to ŷ.& so on. for every row.
based on all these diff. between y & ŷ. we cal cost func.

    C= Σ ½(y-ŷ)sq

![enter image description here](https://github.com/DevMAdi/Artifical--Neural-Network/blob/master/img/Screenshot%20%282783%29.png?raw=true)
once we have full Cost func. we go back & update wt.

Things to note:
	1. we have only 1 perceptron here
	2. wt. is shared by all rows in dataset. which is why we see C= Σ ½(y-ŷ)sq. 		
	3. i.e summation of wt. & then update it.
	4. This complete 1 epoch. we do this again with aim to min. cost func.
		This entire process is called **Backpropagation**.
				
## Gradient descent

for neural n/w to learn- we need Backpropagation.
error/ cost func. is Backpropagated & wt. adjusted.

**How wt. adjusted?**
**How to min. cost func??**

method 1: brute force
![enter image description here](https://github.com/DevMAdi/Artifical--Neural-Network/blob/master/img/Screenshot%20%282785%29.png?raw=true)
	take random wt. get ŷ & cost func. plot them
	best one is at the min. pt.

**Que**. why not try brute force method. why not try 1000 diff i/p wt. & find which is best?
**Ans.** as we increase wt./ synapse. we face curse of dimensionality.
![enter image description here](https://github.com/DevMAdi/Artifical--Neural-Network/blob/master/img/Screenshot%20%282786%29.png?raw=true)
25 weight. 1000 comb. for each weight i.e
	1000 X 1000 X 1000 X ... 25 times = 10 power 75. which can't be calculated

Solution to reduce cost func. without brute force method
**method 2**:	**Gradient descent**

![enter image description here](https://github.com/DevMAdi/Artifical--Neural-Network/blob/master/img/Screenshot%20%282787%29.png?raw=true)
	start at 1 pt. find slope their & follow it
	repeat until minima found

## Stochastic Gradient descent
demerit of grad. desc:
![enter image description here](https://github.com/DevMAdi/Artifical--Neural-Network/blob/master/img/Screenshot%20%282788%29.png?raw=true)
	require cost func. to be **convex** else converge on **local minima** & we wont get optimized wt.

**soln**: 	**Stochastic Gradient descent**

******* Difference *******
![enter image description here](https://github.com/DevMAdi/Artifical--Neural-Network/blob/master/img/Screenshot%20%282789%29.png?raw=true)
normal Grad desc. take all row. plug in neural n/w
	cal. cost func & adjust wt(s).
	This is called grad desc or also called as **Batch Gradient descent.**
	i.e take whole batch from sample, apply & run that

![enter image description here](https://github.com/DevMAdi/Artifical--Neural-Network/blob/master/img/Screenshot%20%282790%29.png?raw=true)
Stochastic Grad desc. on rt. side
	take row 1 by 1. plug in. cal cost func & adjust wt.
	then go to next row. & repeat
	Thus we're adjusting wt. after every single row rather then doing everything together & then adjust wt(case of grad. desc.)

**Stochastic Grad desc:**

> avoid local minima- doing 1 row at time & thus fluctuation is high &
> so very likely to find global minima.
> 
> faster then batch grad desc.

**Batch Gradient descent:**
 - it is **deterministic** algo while Sto Grad desc. is **random**.
 - if we have same starting wt. in batch desc. we will get same
   iteration & result unlike in sto. desc. as row picking at random.
   Thus batch method has merit in this case

**Mini Batch grad Desc:**
 - combine both method
 - instead of entire sample of data or 1 row at time- we take batches of row say 10 or 15 element & then run it.

### 	Training ANN With Stochastic Grad Desc
![enter image description here](https://github.com/DevMAdi/Artifical--Neural-Network/blob/master/img/Screenshot%20%282791%29.png?raw=true)
