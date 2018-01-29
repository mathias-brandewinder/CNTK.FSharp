# CNTK.FSharp

Status: experimental

Goal: provide F# utilities to make the CNTK .NET API pleasant to use from the F# scripting environment

## Contributing

Given the early stage of the project, and the uncertainties around overall design, 
I plan on keeping things fluid at the moment, explore ideas to see what works and what doesn't, 
and keep code primarily in scripts, until things settle down.

I plan on exploring ideas in experimental branches first, and slowly integrate ideas in master. 
Once things stabilize, it will be time for a library with a stable API. 

Ideally, if you have ideas, submit them as an issue, linking to your branch, 
so we can start a discussion! I am also usually around the fsharp.org Slack, in the datascience channel.

## Status/Log

### Jan 28, 2018

Most C# examples have been replicated (see examples folder).

At that point, the `CNTK.Sequential.fsx` has a reasonably working version of sequential models, 
creating a model using a single input variable, by composing layers in a linear fashion, with 
a few basic layers implemented. See `examples/MNIST/MNIST-CNN.Seq.fsx` for an example.

However, sequential models do not cover all use cases, specifically:

- a model could use more than a single input variable,
- a model could fork and join functions.

How to approach that issue is still in flux. The main goal here would be to enable users with more 
advanced scenarios to create their own models. Specific goals are:

- separate expressing the model from specifying what device (CPU or GPU) to run on,
- simplify creating expressions, to limit explicit conversion of `Function` into `Variable`, and 
make models more readable,
- ideally, separate model specification from reading data and training.

Next steps:

1. Refine Sequential, which is likely going to stay there. 

- add missing layers, using the Python examples and Keras as an inspiration,
- convert other sequential models from the Python samples,
- test on Linux, non Windows systems,
- test with GPU,
- try packaging as a library, offering script-friendly utilities.

2. Experiment with a more general modelling approach.

- create an artificial example with 2 input variables, and a model with fork/join of functions
- experiment with other complex models, if available,
- explore whether computation expressions are a viable option to hide `Device` from the creation of a trainable `Function`.

3. Miscellaneous

- explore ways to simplify expressions, one route is to create overloads for most functions in `CNTKLib`,
- dive into data readers

### Nov 3, 2017

Plan: as a first step, focus on replicating the 
[existing C# examples](https://github.com/Microsoft/CNTK/tree/master/Examples/TrainingCSharp), 
to understand better what works and what doesn't. 
