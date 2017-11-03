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

let shape (dims:int seq) = NDShape.CreateNDShape dims

// Creating a synthetic dataset

let generateGaussianNoise (random:Random) (mean, stdDev) =
        
    let u1 = 1.0 - random.NextDouble()
    let u2 = 1.0 - random.NextDouble()
    let stdNormalRandomValue = sqrt(-2.0 * log(u1)) * sin(2.0 * Math.PI * u2)

    mean + stdDev * stdNormalRandomValue
 
let random = Random(0)

let GenerateRawDataSamples(sampleSize,inputDim,numOutputClasses) =
        
    let features = Array.init (sampleSize * inputDim) (fun _ -> float32 0.)
    let oneHotLabels = Array.init (sampleSize * numOutputClasses) (fun _ -> float32 0.)

    for sample in 0 .. sampleSize - 1 do

        let label = random.Next(numOutputClasses)
        for i in 0 .. numOutputClasses - 1 do             
            oneHotLabels.[sample * numOutputClasses + i] <- if label = i then float32 1.0 else float32 0.0
                
        for i in 0 .. inputDim - 1 do               
            features.[sample * inputDim + i] <- float32 (generateGaussianNoise random (3.0, 1.0)) * float32 (label + 1)
            
    features, oneHotLabels
            
let GenerateValueData(sampleSize:int, inputDim:int, numOutputClasses:int, device:DeviceDescriptor) =
        
    let features, oneHotLabels = GenerateRawDataSamples(sampleSize, inputDim, numOutputClasses)
            
    let featureValue = Value.CreateBatch (NDShape.CreateNDShape [ inputDim ], features, device)
    let labelValue = Value.CreateBatch (NDShape.CreateNDShape [ numOutputClasses ], oneHotLabels, device)

    featureValue, labelValue
    
let printTrainingProgress(trainer:Trainer, minibatchIdx:int, outputFrequencyInMinibatches:int) =
        
    if 
        // print out only after every x minibatches
        (minibatchIdx % outputFrequencyInMinibatches) = 0 && 
        trainer.PreviousMinibatchSampleCount() <> (uint32 0)
    then  
        let trainLossValue = trainer.PreviousMinibatchLossAverage() |> float32
        let evaluationValue = trainer.PreviousMinibatchEvaluationAverage() |> float32
        printfn "Minibatch: %i CrossEntropyLoss = %f, EvaluationCriterion = %f" minibatchIdx trainLossValue evaluationValue

// Creating and training the model

let inputDim = 3
let numOutputClasses = 2

let device = DeviceDescriptor.CPUDevice

let featureVariable = Variable.InputVariable(shape[inputDim], DataType.Float)
let labelVariable = Variable.InputVariable(shape[numOutputClasses], DataType.Float)

let createLinearModel(input:Variable, outputDim:int, device:DeviceDescriptor) =
        
    let inputDim = input.Shape.[0]

    let weights = new Parameter(shape [ outputDim; inputDim ], DataType.Float, 1.0, device, "w")
    let bias = new Parameter(shape [ outputDim ], DataType.Float, 0.0, device, "b")
        
    new Variable(CNTKLib.Times(weights, input)) + bias

let classifierOutput = createLinearModel(featureVariable, numOutputClasses, device)
let loss = CNTKLib.CrossEntropyWithSoftmax(new Variable(classifierOutput), labelVariable)
let evalError = CNTKLib.ClassificationError(new Variable(classifierOutput), labelVariable)

let learningRatePerSample = new TrainingParameterScheduleDouble(0.02, uint32 1)

let parameterLearners =
    ResizeArray<Learner>(
        [ 
            Learner.SGDLearner(classifierOutput.Parameters(), learningRatePerSample) 
        ])
        
let trainer = Trainer.CreateTrainer(classifierOutput, loss, evalError, parameterLearners)

let minibatchSize = 64
let numMinibatchesToTrain = 1000
let updatePerMinibatches = 50
        
for minibatchCount in 1 .. (numMinibatchesToTrain) do
        
    let features, labels = GenerateValueData(minibatchSize, inputDim, numOutputClasses, device)
    
    let batch = 
        [ 
            (featureVariable, features) 
            (labelVariable, labels) 
        ] 
        |> dict
            
    trainer.TrainMinibatch(batch, device) |> ignore
            
    printTrainingProgress(trainer, minibatchCount, updatePerMinibatches)
