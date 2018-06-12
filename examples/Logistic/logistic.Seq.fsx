(*
F# port of the original C# example from the CNTK docs:
https://github.com/Microsoft/CNTK/blob/master/Examples/TrainingCSharp/Common/LogisticRegression.cs
*)

#load "../../ScriptLoader.fsx"
open CNTK

#r "../../build/CNTK.FSharp.dll"
open CNTK.FSharp
open CNTK.FSharp.Sequential

open System
open System.Collections.Generic
open System.Text
open System.IO
open System.Threading

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
let generateSyntheticData (inputDim, numOutputClasses) =
        
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

    Seq.initInfinite (fun _ ->
        let labels = random.Next numOutputClasses
        let oneHotLabels = labels |> oneHot numOutputClasses
        let features = labels |> generateFeatures

        features, oneHotLabels
        )

let inputDim = 3
let numOutputClasses = 2

let featureStreamName = "features"
let labelsStreamName = "labels"

let builder = StringBuilder ()
let dataFile = Path.Combine(__SOURCE_DIRECTORY__, "logisticData.txt")

generateSyntheticData (inputDim, numOutputClasses)
|> Seq.take 1000
|> Seq.iter (fun (features,labels) ->
        sprintf "|%s %s |%s %s"
            labelsStreamName
            (labels |> Array.map (sprintf "%.3f") |> String.concat " ")
            featureStreamName
            (features |> Array.map (sprintf "%.3f") |> String.concat " ")
        |> fun line -> builder.AppendLine(line)
        |> ignore
    )
builder.Remove(builder.Length - 2, 2)
File.WriteAllText(dataFile, builder.ToString())

let device = DeviceDescriptor.CPUDevice

let featureVariable = Variable.InputVariable(shape[inputDim], DataType.Float)
let labelVariable = Variable.InputVariable(shape[numOutputClasses], DataType.Float)

let network : Computation = Layer.dense numOutputClasses

let spec = {
    Features = featureVariable
    Labels = labelVariable
    Model = network
    Loss = CrossEntropyWithSoftmax
    Eval = ClassificationError
    }
    
let config = {
    MinibatchSize = 64
    Epochs = 10
    Device = device
    Schedule = { Rate = 0.01; MinibatchSize = 1 }
    Optimizer = SGD
    CancellationToken = CancellationToken.None
    }

let streamConfigurations = 
    [|
        new StreamConfiguration(featureStreamName, inputDim)
        new StreamConfiguration(labelsStreamName, numOutputClasses)
    |]

let minibatchSource = 
    MinibatchSource.TextFormatMinibatchSource(
        dataFile, 
        streamConfigurations, 
        MinibatchSource.FullDataSweep)

let learner = Learner ()
learner.MinibatchProgress.Add Minibatch.basicPrint

let predictor = learner.learn minibatchSource (featureStreamName, labelsStreamName) config spec

let modelFile = Path.Combine(__SOURCE_DIRECTORY__, "logistic.model")
predictor.Save(modelFile)

let ValidateModelWithMinibatchSource(
    modelFile:string, 
    testMinibatchSource:MinibatchSource,
    featureInputName:string, 
    labelInputName:string, 
    device:DeviceDescriptor, 
    maxCount:int
    ) =

        let model : Function = Function.Load(modelFile, device)
        let imageInput = model.Arguments.[0]
        let labelOutput = model.Output

        let featureStreamInfo = testMinibatchSource.StreamInfo(featureInputName)
        let labelStreamInfo = testMinibatchSource.StreamInfo(labelInputName)

        let batchSize = 50

        let rec countErrors (total,errors) =

            printfn "Total: %i; Errors: %i" total errors

            let minibatchData = testMinibatchSource.GetNextMinibatch((uint32)batchSize, device)

            if (minibatchData = null || minibatchData.Count = 0)
            then (total,errors)        
            else

                let total = total + minibatchData.[featureStreamInfo].numberOfSamples

                // find the index of the largest label value
                let labelData = minibatchData.[labelStreamInfo].data.GetDenseData<float32>(labelOutput)
                let expectedLabels = 
                    labelData 
                    |> Seq.map (fun l ->                         
                        let largest = l |> Seq.max
                        l.IndexOf largest
                        )

                let inputDataMap = 
                    [
                        imageInput, minibatchData.[featureStreamInfo].data
                    ]
                    |> dataMap

                let outputDataMap = 
                    [ 
                        labelOutput, null 
                    ] 
                    |> dataMap
                    
                model.Evaluate(inputDataMap, outputDataMap, device)

                let outputData = outputDataMap.[labelOutput].GetDenseData<float32>(labelOutput)
                let actualLabels =
                    outputData 
                    |> Seq.map (fun l ->                         
                        let largest = l |> Seq.max
                        l.IndexOf largest
                        )

                let misMatches = 
                    (actualLabels,expectedLabels)
                    ||> Seq.zip
                    |> Seq.sumBy (fun (a, b) -> if a = b then 0 else 1)

                let errors = errors + misMatches

                if (int total > maxCount)
                then (total,errors)
                else countErrors (total,errors)

        countErrors (uint32 0,0)

let total,errors = 
    ValidateModelWithMinibatchSource(
        modelFile, 
        minibatchSource,
        featureStreamName, 
        labelsStreamName, 
        DeviceDescriptor.CPUDevice,
        1000)

printfn "Total: %i / Errors: %i" total errors 
