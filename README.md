# SGD-on-Boston-Dataset
comparison between manual implementation of linear regression and SGD regressor of sklearn on Boston home price dataset

## Linear Regression:-

When we saw the word regression in Machine Learning field ,what comes in our mind .A regression problem is when output variable is a real or continuous value , such as ‘price’ , ‘salary’, ‘weight’ .There are many regression model or algorithm but simplest one is Linear Regression.
It tries to fit best hyper-plane which goes through the points 



The big question is how we will find our best hyper-plane ?
You don’t need to know any statistics our linear algebra to understand linear regression.Let me give some intuition behind linear regression.
Suppose that you have found your best hyper-plane but there may be bunch of points which are lying outside  the hyper-plane .Let’s take an example you estimated a point which is lying on the plane but actually it was above the plane ,that makes an error so for this we have to minimise the error for all those points which are lying above or down the plane.Let’s take a point p1 which actually lies above the plane and point p2 which actually lies below the plane ,the actual y coordinates of p1 and p2 are y1 and y2 respectively.Then

error1=y1(actual)-y1(predicted) > 0 since y1(actual) lies above to the plane

error2=y2(actual)-y2(predicted)<0 since y2(actual) lies below the plane.



So this problem becomes optimization problem as we have seen in the case of Logistic Regression. In this problem we have to find that hyperplane which minimise the errors for all points .

We know that equation of plane is Wx+b=0 where b is intercept ,our task is to find best W and b.

Optimization equation:-

                         
best (W , b) =1/n(argmin Σ (y(actual)-y(pred))^2)
                       
                        
Since distance between actual and predicted may be positive or negative so we squared it. As in the case of Logistic regression there is a logistic loss and here in linear regression there is squared loss ,from above equation we can see that square loss occur in the case of linear regression.

In the real world data is not perfect so we have to add some error that leads to regularisation ,it forces W to be small as possible so equation becomes



                         
best (W , b) =1/n(argmin Σ (y(actual)-y(pred))^2) + regularisation
                       


## BOSTON DATASET:-

### Data description
The Boston data frame has 506 rows and 14 columns.
This data frame contains the following columns:
crim
per capita crime rate by town.
zn
proportion of residential land zoned for lots over 25,000 sq.ft.
indus
proportion of non-retail business acres per town.
chas
Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
nox
nitrogen oxides concentration (parts per 10 million).
rm
average number of rooms per dwelling.
age
proportion of owner-occupied units built prior to 1940.
dis
weighted mean of distances to five Boston employment centres.
rad
index of accessibility to radial highways.
tax
full-value property-tax rate per $10,000.
ptratio
pupil-teacher ratio by town.
black
1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.
lstat
lower status of the population (percent).
medv
median value of owner-occupied homes in $1000s.

## Gradient Descent: Algorithm-
 
It is a first order iterative optimization algorithm for finding minimum of a function .To find a local minima of a function using gradient descent ,one takes steps proportional to the negative of gradient of the function at the curent point .If instead , one takes steps proportional to the positive of the gradient approaches local maxima of a function.

Our equation for linear regression :-
            
L(W,B)= 1/n Σ (y-w.x-b)^2
          
           
take partial derivative w.r.t W and B

           
dL/dW = 1/n Σ (-2x)(y-w.x-b)
           
           
          
dL/dB= 1/n Σ (-2)(y-w.x-b)
         


W(j+1)=W(j) – r *(dL/dW)


B(j+1)=B(j) – r*(dL/dB)


iterate this till W(k) becomes nearly equal to W(k+1) and B(k) nearly equal to B(k+1).


After that find out predicted output on test data and calculate mean squared error.
