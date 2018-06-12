(*
F# port of the original C# example from the CNTK docs:
https://github.com/Microsoft/CNTK/master/Examples/TrainingCSharp/Common/LSTMSequenceClassifier.cs
*)

#load "../../ScriptLoader.fsx"
open CNTK

#r "../../build/CNTK.FSharp.dll"
open CNTK.FSharp
open CNTK.FSharp.Sequential

open System.IO
open System.Collections.Generic
open System.Threading

let DataFolder = __SOURCE_DIRECTORY__

let device = DeviceDescriptor.CPUDevice

let inputDim = 2000
let cellDim = 25
let hiddenDim = 25
let embeddingDim = 50
let numOutputClasses = 5

// build the model
let featureStreamName = "features"
let features = Variable.InputVariable(shape [ inputDim ], DataType.Float, featureStreamName, null, true)
let labelStreamName = "labels"
let labels = Variable.InputVariable(shape [ numOutputClasses ], DataType.Float, labelStreamName, ResizeArray<Axis>( [ Axis.DefaultBatchAxis() ]), true)

let classifierOutput =
    Recurrent.LSTMSequenceClassifierNet numOutputClasses embeddingDim hiddenDim cellDim

let spec = {
    Features = features
    Labels = labels
    Model = classifierOutput
    Loss = CrossEntropyWithSoftmax
    Eval = ClassificationError
    }

// prepare training data
let streamConfigurations = 
    [|
        new StreamConfiguration(featureStreamName, inputDim, true, "x")
        new StreamConfiguration(labelStreamName, numOutputClasses, false, "y")
    |]

let minibatchSource = 
    MinibatchSource.TextFormatMinibatchSource(
        Path.Combine(DataFolder, "Train.ctf"), 
        streamConfigurations, 
        MinibatchSource.InfinitelyRepeat, 
        true
        )

let featureStreamInfo = minibatchSource.StreamInfo(featureStreamName)
let labelStreamInfo = minibatchSource.StreamInfo(labelStreamName)

// set per sample learning rate
let config = {
    MinibatchSize = 200
    Epochs = 5
    Device = DeviceDescriptor.CPUDevice
    Schedule = { Rate = 0.0005; MinibatchSize = 1 }
    Optimizer = MomentumSGD 256.
    CancellationToken = CancellationToken.None
    }

let trainer = Learner ()
trainer.MinibatchProgress.Add(Minibatch.basicPrint)

let predictor = trainer.learn minibatchSource (featureStreamName,labelStreamName) config spec
let modelFile = Path.Combine(__SOURCE_DIRECTORY__,"LSTM.model")

predictor.Save(modelFile)