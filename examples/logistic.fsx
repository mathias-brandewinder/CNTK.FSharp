(*
F# port of the original C# example from the CNTK docs:
https://github.com/Microsoft/CNTK/blob/master/Examples/TrainingCSharp/Common/LogisticRegression.cs
*)

// Use the CNTK.fsx file to load the dependencies.

#load "../CNTK.fsx"
open CNTK

open System
open System.Collections.Generic

// Conversion of the original C# code to an F# script

// Helpers to simplify model creation from F#

// Creating a synthetic dataset

let generateGaussianNoise (random:Random) (mean, stdDev) =
        
    let u1 = 1.0 - random.NextDouble()
    let u2 = 1.0 - random.NextDouble()
    let stdNormalRandomValue = sqrt(-2.0 * log(u1)) * sin(2.0 * Math.PI * u2)

    mean + stdDev * stdNormalRandomValue
 
let random = Random(0)

// generate synthetic data: each group is normally distributed,
// centered around (3,3), (6,6), ... 
// this is not as efficient as the original example, but clearer: 
// we create an array of labels first, which we then transform
// into one-hot form, and into its features. 
let generateSyntheticData(sampleSize,inputDim,numOutputClasses) =
        
    // utility one-hot encoder
    let oneHot classes value = 
        Array.init classes (fun i -> 
            if i = value then float32 1. else float32 0.
            )
    
    // generate synthetic feature for given label
    let generateFeatures label =
        Array.init inputDim (fun _ -> 
            generateGaussianNoise random (3.0, 1.0) * (float label + 1.) |> float32
            )

    let labels = Array.init sampleSize (fun _ -> random.Next numOutputClasses)
    let oneHotLabels = labels |> Array.collect (oneHot numOutputClasses)
    let features = labels |> Array.collect (generateFeatures)

    features, oneHotLabels

let GenerateValueData(sampleSize:int, inputDim:int, numOutputClasses:int, device:DeviceDescriptor) =
        
    let features, oneHotLabels = generateSyntheticData(sampleSize, inputDim, numOutputClasses)
            
    let featureValue = Value.CreateBatch (shape [ inputDim ], features, device)
    let labelValue = Value.CreateBatch (shape [ numOutputClasses ], oneHotLabels, device)

    featureValue, labelValue
    
// Creating and training the model

let inputDim = 3
let numOutputClasses = 2


let predictor : Layer = Layers.dense numOutputClasses

type Loss = 
    | CrossEntropyWithSoftmax
    | ClassificationError

let evalutation (loss:Loss) (predicted:Function,actual:Variable) =
    match loss with
    | CrossEntropyWithSoftmax -> CNTKLib.CrossEntropyWithSoftmax(new Variable(predicted),actual)
    | ClassificationError -> CNTKLib.ClassificationError(new Variable(predicted),actual)

type Schedule = {
    Rate:float
    MinibatchSize:int
    }

type Specification = {
    Features: Variable
    Labels: Variable
    Model: Layer
    Loss: Loss
    Eval: Loss
    Schedule: Schedule
    }

let featureVariable = Variable.InputVariable(shape[inputDim], DataType.Float)
let labelVariable = Variable.InputVariable(shape[numOutputClasses], DataType.Float)

let schedule = { Rate = 0.02; MinibatchSize = 1 }

let specification = {
    Features = featureVariable
    Labels = labelVariable
    Model = predictor
    Loss = CrossEntropyWithSoftmax
    Eval = ClassificationError
    Schedule = schedule
    }

let learning (predictor:Function) (schedule:Schedule) =   
    let learningRatePerSample = 
        new TrainingParameterScheduleDouble(schedule.Rate, uint32 schedule.MinibatchSize)
    let parameterLearners =
        ResizeArray<Learner>(
            [ 
                Learner.SGDLearner(predictor.Parameters(), learningRatePerSample) 
            ])
    parameterLearners
    
let createTrainer (device:DeviceDescriptor) (spec:Specification) =

    let predictor = spec.Model device spec.Features

    let loss = evalutation spec.Loss (predictor,spec.Labels)

    let eval = evalutation spec.Eval (predictor,spec.Labels)
    
    let parameterLearners = learning predictor spec.Schedule
      
    Trainer.CreateTrainer(predictor, loss, eval, parameterLearners)

let device = DeviceDescriptor.CPUDevice

let trainer = createTrainer device specification
let minibatchSize = 64
let numMinibatchesToTrain = 1000
let updatePerMinibatches = 50

let report = progress (trainer, updatePerMinibatches) 

for minibatchCount in 1 .. (numMinibatchesToTrain) do
        
    let features, labels = GenerateValueData(minibatchSize, inputDim, numOutputClasses, device)
    
    let batch = 
        [ 
            (featureVariable, features) 
            (labelVariable, labels) 
        ] 
        |> dict
            
    trainer.TrainMinibatch(batch, device) |> ignore
            
    report minibatchCount |> printer

// classifierOutput |> Debug.valueAt [ 1.0f; 2.0f;  3.0f] 
