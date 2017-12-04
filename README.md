# CNTK.FSharp

## experiment-interpreter branch

Goal: explore modelling [CNTK expressions using a DU wrapper](https://github.com/mathias-brandewinder/CNTK.FSharp/blob/experiment-interpreter/CNTK.fsx#L61-L76).  

- Replicated Logistic and a good part of MNIST-MLP.
- Attempted to wrap the construction of a Trainer, passing in loss and eval functions as a DU marker, 
and returning a ready-to-train Trainer.

Conclusions:  
- Requires re-implementing how shapes / dimensions are transformed by functions
- Enables defining a computation graph independently from the device it will run on
- Enables deferring how to name Variables/Functions
- Not sure how to handle Scalars, in particular for dimensions.
